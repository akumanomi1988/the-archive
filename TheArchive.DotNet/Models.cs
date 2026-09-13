using System.Text.Json;
using System.Text.Json.Serialization;

namespace TheArchive;

public sealed record Book(long Id, string Title, string Author, string? Language, string Path,
    long SizeBytes, string ModifiedIso, string Sha256, string? Enriched, string EnrichmentStatus,
    string? Genre, int? Year, string? CreatedAt, string? UpdatedAt);

public sealed record BookState(long BookId, bool Favorite, bool Read, bool Pending,
    string? LastMode, long? ScrollTop, int? Page, double? Percent, string? UpdatedAt);

public sealed class ProgressRequest
{
    public string? Mode { get; set; }
    [JsonPropertyName("scroll_top")]
    public long? ScrollTop { get; set; }
    public int? Page { get; set; }
    public double? Percent { get; set; }
}

public sealed class StateRequest
{
    public bool? Favorite { get; set; }
    public bool? Read { get; set; }
    public bool? Pending { get; set; }
}

public sealed class ReindexRequest { public string Mode { get; set; } = "sync"; }
public sealed class EnrichBatchRequest { public long[]? Ids { get; set; } public string? Language { get; set; } }
public sealed class EnrichAllRequest
{
    [JsonPropertyName("only_pending")]
    public bool OnlyPending { get; set; } = true;
    [JsonPropertyName("only_never")]
    public bool OnlyNever { get; set; }
    [JsonPropertyName("throttle_ms")]
    public int ThrottleMs { get; set; }
    public int? Limit { get; set; }
    public string? Language { get; set; }
}

public static class JsonData
{
    public static object? Parse(string? value)
    {
        if (string.IsNullOrWhiteSpace(value)) return null;
        try { return JsonSerializer.Deserialize<JsonElement>(value); } catch { return value; }
    }
}
