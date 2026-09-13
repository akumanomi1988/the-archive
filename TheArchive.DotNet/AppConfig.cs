using System.Text.Json;
using System.Text.Json.Serialization;

namespace TheArchive;

public sealed class AppConfig
{
    [JsonPropertyName("library_path")] public string LibraryPath { get; set; } = "./biblioteca";
    [JsonPropertyName("db_path")] public string DbPath { get; set; } = "./skald.db";
    [JsonPropertyName("metadata_provider")] public string MetadataProvider { get; set; } = "openlibrary";
    [JsonPropertyName("metadata_api_url")] public string MetadataApiUrl { get; set; } = "https://openlibrary.org/search.json";
    [JsonPropertyName("metadata_timeout_seconds")] public double MetadataTimeoutSeconds { get; set; } = 20;
    [JsonPropertyName("metadata_min_match_score")] public double MetadataMinMatchScore { get; set; } = 0.55;
    [JsonPropertyName("metadata_throttle_ms")] public int MetadataThrottleMs { get; set; } = 1100;
    [JsonPropertyName("cors_origins")] public string[] CorsOrigins { get; set; } = ["*"];
    [JsonPropertyName("page_size_default")] public int PageSizeDefault { get; set; } = 20;
    [JsonPropertyName("enrichment_batch_size")] public int EnrichmentBatchSize { get; set; } = 10;
    [JsonPropertyName("default_language")] public string DefaultLanguage { get; set; } = "es";
    [JsonPropertyName("open_browser")] public bool OpenBrowser { get; set; } = true;
    [JsonPropertyName("port")] public int Port { get; set; } = 8000;

    [JsonExtensionData] public Dictionary<string, JsonElement>? Extra { get; set; }

    public static (AppConfig Config, string Path) Load(string root)
    {
        var requested = Environment.GetEnvironmentVariable("SKALD_CONFIG");
        var path = string.IsNullOrWhiteSpace(requested) ? System.IO.Path.Combine(root, "config.json") : System.IO.Path.GetFullPath(requested);
        AppConfig config;
        if (File.Exists(path))
            config = JsonSerializer.Deserialize<AppConfig>(File.ReadAllText(path), JsonOptions()) ?? new();
        else
            config = new();
        var baseDir = System.IO.Path.GetDirectoryName(path) ?? root;
        config.LibraryPath = Resolve(baseDir, config.LibraryPath);
        config.DbPath = Resolve(baseDir, config.DbPath);
        if (int.TryParse(Environment.GetEnvironmentVariable("THE_ARCHIVE_PORT"), out var port) && port is > 0 and <= 65535)
            config.Port = port;
        return (config, path);
    }

    public void Save(string path)
    {
        Directory.CreateDirectory(System.IO.Path.GetDirectoryName(path)!);
        File.WriteAllText(path, JsonSerializer.Serialize(this, JsonOptions(true)));
    }

    public static JsonSerializerOptions JsonOptions(bool indented = false) => new(JsonSerializerDefaults.Web)
    {
        WriteIndented = indented,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    private static string Resolve(string root, string path) => System.IO.Path.GetFullPath(System.IO.Path.IsPathRooted(path) ? path : System.IO.Path.Combine(root, path));
}
