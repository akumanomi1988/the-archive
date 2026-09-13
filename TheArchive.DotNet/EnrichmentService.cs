using System.Globalization;
using System.Text;
using System.Text.Json;

namespace TheArchive;

public sealed class EnrichmentService
{
    private const string Fields = "key,title,author_name,first_publish_year,subject,cover_i,language,first_sentence,ratings_average,edition_count";
    private readonly Func<AppConfig> _config;
    private readonly IHttpClientFactory _http;
    private readonly SemaphoreSlim _requestGate = new(1, 1);
    private DateTimeOffset _lastRequest = DateTimeOffset.MinValue;

    public EnrichmentService(Func<AppConfig> config, IHttpClientFactory http)
    {
        _config = config;
        _http = http;
    }

    public async Task<(string Status, string Json, string Source, string? Error)> Enrich(Book book, string? language, CancellationToken ct = default)
    {
        var cfg = _config();
        if (!string.Equals(cfg.MetadataProvider, "openlibrary", StringComparison.OrdinalIgnoreCase))
            return ("failed", "null", cfg.MetadataProvider, $"Proveedor de metadatos no compatible: {cfg.MetadataProvider}");

        await _requestGate.WaitAsync(ct);
        try
        {
            var remaining = TimeSpan.FromMilliseconds(Math.Max(0, cfg.MetadataThrottleMs)) - (DateTimeOffset.UtcNow - _lastRequest);
            if (remaining > TimeSpan.Zero) await Task.Delay(remaining, ct);

            var query = new List<string> { $"title={Uri.EscapeDataString(book.Title)}" };
            if (!IsUnknown(book.Author)) query.Add($"author={Uri.EscapeDataString(book.Author)}");
            query.Add($"fields={Uri.EscapeDataString(Fields)}");
            query.Add("limit=5");
            if (!string.IsNullOrWhiteSpace(language)) query.Add($"lang={Uri.EscapeDataString(language[..Math.Min(2, language.Length)])}");
            var separator = cfg.MetadataApiUrl.Contains('?') ? "&" : "?";
            var url = cfg.MetadataApiUrl + separator + string.Join("&", query);

            var client = _http.CreateClient();
            client.Timeout = TimeSpan.FromSeconds(Math.Clamp(cfg.MetadataTimeoutSeconds, 2, 120));
            client.DefaultRequestHeaders.UserAgent.ParseAdd("TheArchive/1.0 (local personal library; metadata lookup)");
            using var response = await client.GetAsync(url, ct);
            _lastRequest = DateTimeOffset.UtcNow;
            response.EnsureSuccessStatusCode();

            using var payload = JsonDocument.Parse(await response.Content.ReadAsStringAsync(ct));
            if (!payload.RootElement.TryGetProperty("docs", out var docs) || docs.ValueKind != JsonValueKind.Array)
                return ("failed", "null", "openlibrary", "Open Library devolvió una respuesta sin resultados");

            JsonElement? best = null;
            var bestScore = 0d;
            foreach (var doc in docs.EnumerateArray())
            {
                var score = MatchScore(book, doc);
                if (score > bestScore) { best = doc.Clone(); bestScore = score; }
            }
            if (best is null || bestScore < Math.Clamp(cfg.MetadataMinMatchScore, 0, 1))
                return ("failed", "null", "openlibrary", $"No se encontró una coincidencia fiable (mejor puntuación: {bestScore:0.00})");

            var item = best.Value;
            var subjects = Strings(item, "subject").Where(IsUsefulSubject).Distinct(StringComparer.OrdinalIgnoreCase).Take(20).ToArray();
            var key = String(item, "key");
            var coverId = Integer(item, "cover_i");
            var requestedLanguage = language ?? book.Language ?? cfg.DefaultLanguage;
            var result = new
            {
                genre = PickGenre(subjects),
                year = Integer(item, "first_publish_year"),
                tags = subjects.Take(8).ToArray(),
                premise = PickSentence(Strings(item, "first_sentence"), requestedLanguage)
                    ?? $"Ficha bibliográfica de {String(item, "title") ?? book.Title}, de {string.Join(", ", Strings(item, "author_name"))}.",
                audience = PickAudience(subjects),
                confidence = Math.Round(bestScore, 2),
                source_title = String(item, "title"),
                source_authors = Strings(item, "author_name"),
                languages = Strings(item, "language").Take(12).ToArray(),
                ratings_average = Number(item, "ratings_average"),
                edition_count = Integer(item, "edition_count"),
                openlibrary_key = key,
                openlibrary_url = key is null ? null : "https://openlibrary.org" + (key.StartsWith('/') ? key : "/works/" + key),
                cover_url = coverId is null ? null : $"https://covers.openlibrary.org/b/id/{coverId}-L.jpg?default=false",
                enriched_by = "openlibrary",
                enriched_at = DateTimeOffset.UtcNow
            };
            return ("ok", JsonSerializer.Serialize(result, AppConfig.JsonOptions()), "openlibrary", null);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            return ("failed", "null", "openlibrary", "La consulta a Open Library agotó el tiempo de espera");
        }
        catch (Exception ex)
        {
            return ("failed", "null", "openlibrary", ex.Message);
        }
        finally { _requestGate.Release(); }
    }

    private static double MatchScore(Book book, JsonElement doc)
    {
        var title = Similarity(book.Title, String(doc, "title"));
        var authors = Strings(doc, "author_name");
        var author = IsUnknown(book.Author) ? 1 : authors.Select(a => Similarity(book.Author, a)).DefaultIfEmpty(0).Max();
        return Math.Clamp(title * 0.7 + author * 0.3, 0, 1);
    }

    private static double Similarity(string? expected, string? actual)
    {
        var a = Normalize(expected); var b = Normalize(actual);
        if (a.Length == 0 || b.Length == 0) return 0;
        if (a == b) return 1;
        if (a.Contains(b) || b.Contains(a)) return 0.82;
        var left = a.Split(' ', StringSplitOptions.RemoveEmptyEntries).ToHashSet();
        var right = b.Split(' ', StringSplitOptions.RemoveEmptyEntries).ToHashSet();
        return (double)left.Intersect(right).Count() / Math.Max(1, left.Union(right).Count());
    }

    private static string Normalize(string? value)
    {
        if (string.IsNullOrWhiteSpace(value)) return "";
        var decomposed = value.Normalize(NormalizationForm.FormD);
        var result = new StringBuilder();
        foreach (var c in decomposed)
            if (CharUnicodeInfo.GetUnicodeCategory(c) != UnicodeCategory.NonSpacingMark)
                result.Append(char.IsLetterOrDigit(c) ? char.ToLowerInvariant(c) : ' ');
        return string.Join(' ', result.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries));
    }

    private static bool IsUnknown(string? value) => string.IsNullOrWhiteSpace(value) || Normalize(value) is "desconocido" or "unknown" or "autor desconocido";
    private static string? String(JsonElement item, string name) => item.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;
    private static int? Integer(JsonElement item, string name) => item.TryGetProperty(name, out var value) && value.TryGetInt32(out var number) ? number : null;
    private static double? Number(JsonElement item, string name) => item.TryGetProperty(name, out var value) && value.TryGetDouble(out var number) ? number : null;
    private static string[] Strings(JsonElement item, string name)
    {
        if (!item.TryGetProperty(name, out var value)) return [];
        if (value.ValueKind == JsonValueKind.String) return [value.GetString()!];
        return value.ValueKind == JsonValueKind.Array ? value.EnumerateArray().Where(x => x.ValueKind == JsonValueKind.String).Select(x => x.GetString()!).Where(x => !string.IsNullOrWhiteSpace(x)).ToArray() : [];
    }

    private static bool IsUsefulSubject(string subject)
    {
        var text = subject.Trim();
        var normalized = Normalize(text);
        if (text.Length is < 2 or > 80 || text.All(c => char.IsDigit(c) || char.IsPunctuation(c) || char.IsWhiteSpace(c))) return false;
        if (normalized.EndsWith("language books") || normalized.EndsWith("language materials") || normalized.StartsWith("translations") || normalized is "texts" or "in english" or "language study") return false;
        return true;
    }

    private static string PickGenre(IEnumerable<string> subjects)
    {
        var candidates = new (string Genre, string[] Terms)[]
        {
            ("Ciencia ficción", ["science fiction", "ciencia ficcion"]), ("Fantasía", ["fantasy", "fantasia"]),
            ("Misterio", ["mystery", "detective", "crime fiction", "misterio"]), ("Romance", ["romance", "love stories"]),
            ("Terror", ["horror", "terror"]), ("Biografía", ["biography", "autobiography", "biografia"]),
            ("Historia", ["history", "historia"]), ("Poesía", ["poetry", "poesia"]),
            ("Teatro", ["drama", "plays", "teatro"]), ("Ensayo", ["essays", "ensayo"]),
            ("Ficción", ["fiction", "novel", "novela", "ficcion"])
        };
        var normalized = subjects.Select(Normalize).ToArray();
        foreach (var candidate in candidates)
            if (normalized.Any(s => candidate.Terms.Any(t => s.Contains(Normalize(t))))) return candidate.Genre;
        return "Desconocido";
    }

    private static string PickAudience(IEnumerable<string> subjects)
    {
        var text = Normalize(string.Join(' ', subjects));
        if (new[] { "juvenile", "young adult", "children", "ninos", "infantil" }.Any(text.Contains)) return "juvenil";
        return "general";
    }

    private static string? PickSentence(IEnumerable<string> sentences, string language)
    {
        var list = sentences.Where(s => !string.IsNullOrWhiteSpace(s)).ToArray();
        if (list.Length == 0) return null;
        var markers = (language.Length >= 2 ? language[..2].ToLowerInvariant() : "es") switch
        {
            "es" => new[] { " el ", " la ", " los ", " las ", " que ", " había ", " años " },
            "fr" => new[] { " le ", " la ", " les ", " des ", " que " },
            "de" => new[] { " der ", " die ", " das ", " und " },
            "pt" => new[] { " o ", " a ", " os ", " as ", " que " },
            _ => new[] { " the ", " a ", " and ", " that " }
        };
        return list.OrderByDescending(s => markers.Count(m => (" " + s.ToLowerInvariant() + " ").Contains(m))).First();
    }
}
