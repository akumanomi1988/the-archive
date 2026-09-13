using Microsoft.Data.Sqlite;
using System.Globalization;
using System.Text;

namespace TheArchive;

public sealed class ArchiveDb
{
    private readonly Func<AppConfig> _config;
    public ArchiveDb(Func<AppConfig> config) => _config = config;
    private SqliteConnection Open()
    {
        var path = _config().DbPath;
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var c = new SqliteConnection(new SqliteConnectionStringBuilder { DataSource = path, Mode = SqliteOpenMode.ReadWriteCreate }.ToString());
        c.Open();
        using var pragma = c.CreateCommand(); pragma.CommandText = "PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;"; pragma.ExecuteNonQuery();
        return c;
    }

    public void Initialize()
    {
        using var c = Open(); using var cmd = c.CreateCommand();
        cmd.CommandText = """
        CREATE TABLE IF NOT EXISTS book (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, author TEXT NOT NULL, language TEXT, path TEXT NOT NULL, size_bytes INTEGER NOT NULL, modified_iso TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE, enriched JSON, enrichment_status TEXT NOT NULL DEFAULT 'none', genre TEXT, year INTEGER, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS bookstate (id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL UNIQUE, favorite_flag INTEGER NOT NULL DEFAULT 0, read_flag INTEGER NOT NULL DEFAULT 0, pending_flag INTEGER NOT NULL DEFAULT 0, last_mode TEXT, scroll_top INTEGER, page INTEGER, percent REAL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS author (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE);
        CREATE TABLE IF NOT EXISTS genre (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE);
        CREATE TABLE IF NOT EXISTS bookauthor (id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL, author_id INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS bookgenre (id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER NOT NULL, genre_id INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS ix_book_title ON book(title); CREATE INDEX IF NOT EXISTS ix_book_author ON book(author); CREATE INDEX IF NOT EXISTS ix_book_sha256 ON book(sha256);
        CREATE INDEX IF NOT EXISTS ix_book_genre ON book(genre); CREATE INDEX IF NOT EXISTS ix_book_year ON book(year);
        """;
        cmd.ExecuteNonQuery();
    }

    public (long Total, List<object> Items) Search(string? q, string? autor, string? titulo, string? genre, int? yearFrom, int? yearTo, string? stateFilter, string? sortBy, int page, int pageSize)
    {
        using var c = Open();
        var where = new List<string>(); var args = new Dictionary<string, object?>();
        void Like(string column, string name, string? value) { if (!string.IsNullOrWhiteSpace(value)) { where.Add($"{column} LIKE @{name} COLLATE NOCASE"); args[name] = $"%{value.Trim()}%"; } }
        if (!string.IsNullOrWhiteSpace(q)) { where.Add("(b.title LIKE @q COLLATE NOCASE OR b.author LIKE @q COLLATE NOCASE)"); args["q"] = $"%{q.Trim()}%"; }
        Like("b.author", "autor", autor); Like("b.title", "titulo", titulo); Like("b.genre", "genre", genre);
        if (yearFrom is not null) { where.Add("b.year >= @yearFrom"); args["yearFrom"] = yearFrom; }
        if (yearTo is not null) { where.Add("b.year <= @yearTo"); args["yearTo"] = yearTo; }
        var stateColumn = stateFilter switch { "favorite" => "favorite_flag", "read" => "read_flag", "pending" => "pending_flag", _ => null };
        if (stateColumn is not null) where.Add($"COALESCE(s.{stateColumn},0)=1");
        var w = where.Count == 0 ? "" : " WHERE " + string.Join(" AND ", where);
        using var count = c.CreateCommand(); count.CommandText = "SELECT COUNT(*) FROM book b LEFT JOIN bookstate s ON s.book_id=b.id" + w; Add(count, args);
        var total = (long)(count.ExecuteScalar() ?? 0L);
        var order = sortBy switch { "author" => "b.author", "year" => "b.year DESC", "added" => "b.created_at DESC", "genre" => "b.genre", "progress" => "s.percent DESC", _ => "b.title" };
        using var cmd = c.CreateCommand();
        cmd.CommandText = $"SELECT b.*, COALESCE(s.favorite_flag,0),COALESCE(s.read_flag,0),COALESCE(s.pending_flag,0),COALESCE(s.percent,0) FROM book b LEFT JOIN bookstate s ON s.book_id=b.id{w} ORDER BY {order} LIMIT @limit OFFSET @offset";
        Add(cmd, args); cmd.Parameters.AddWithValue("@limit", pageSize); cmd.Parameters.AddWithValue("@offset", (page - 1) * pageSize);
        var items = new List<object>(); using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            var b = ReadBook(r); var enriched = JsonData.Parse(b.Enriched); object tags = Array.Empty<string>(); object? wordCount = null;
            if (enriched is System.Text.Json.JsonElement e && e.ValueKind == System.Text.Json.JsonValueKind.Object) { if (e.TryGetProperty("tags", out var t)) tags = t; if (e.TryGetProperty("word_count", out var wc)) wordCount = wc; }
            items.Add(new { id=b.Id,title=b.Title,author=b.Author,language=b.Language,size_bytes=b.SizeBytes,modified_iso=b.ModifiedIso,sha256=b.Sha256,genre=b.Genre,year=b.Year,enrichment_status=b.EnrichmentStatus,favorite=r.GetInt64(14)!=0,read=r.GetInt64(15)!=0,pending=r.GetInt64(16)!=0,progress_percent=Math.Round(r.GetDouble(17),1),tags,word_count=wordCount,created_at=b.CreatedAt,cover_url=$"/books/{b.Id}/cover/thumb" });
        }
        return (total, items);
    }

    public Book? GetBook(long id)
    {
        using var c=Open(); using var cmd=c.CreateCommand(); cmd.CommandText="SELECT * FROM book WHERE id=@id"; cmd.Parameters.AddWithValue("@id",id); using var r=cmd.ExecuteReader(); return r.Read()?ReadBook(r):null;
    }

    public BookState GetState(long id)
    {
        using var c=Open(); using var cmd=c.CreateCommand(); cmd.CommandText="SELECT book_id,favorite_flag,read_flag,pending_flag,last_mode,scroll_top,page,percent,updated_at FROM bookstate WHERE book_id=@id";cmd.Parameters.AddWithValue("@id",id);using var r=cmd.ExecuteReader();
        return r.Read()?new(id,r.GetInt64(1)!=0,r.GetInt64(2)!=0,r.GetInt64(3)!=0,N(r,4),NL(r,5),NI(r,6),ND(r,7),N(r,8)):new(id,false,false,false,null,null,null,null,null);
    }

    public BookState SaveState(long id, StateRequest? state=null, ProgressRequest? progress=null)
    {
        var old=GetState(id); var now=DateTimeOffset.UtcNow.ToString("O");
        using var c=Open();using var cmd=c.CreateCommand();
        cmd.CommandText="INSERT INTO bookstate(book_id,favorite_flag,read_flag,pending_flag,last_mode,scroll_top,page,percent,updated_at) VALUES(@id,@f,@r,@p,@m,@s,@pg,@pc,@u) ON CONFLICT(book_id) DO UPDATE SET favorite_flag=@f,read_flag=@r,pending_flag=@p,last_mode=@m,scroll_top=@s,page=@pg,percent=@pc,updated_at=@u";
        cmd.Parameters.AddWithValue("@id",id);cmd.Parameters.AddWithValue("@f",state?.Favorite??old.Favorite);cmd.Parameters.AddWithValue("@r",state?.Read??old.Read);cmd.Parameters.AddWithValue("@p",state?.Pending??old.Pending);
        cmd.Parameters.AddWithValue("@m",(object?)(progress?.Mode??old.LastMode)??DBNull.Value);cmd.Parameters.AddWithValue("@s",(object?)(progress?.ScrollTop??old.ScrollTop)??DBNull.Value);cmd.Parameters.AddWithValue("@pg",(object?)(progress?.Page??old.Page)??DBNull.Value);cmd.Parameters.AddWithValue("@pc",(object?)(progress?.Percent??old.Percent)??DBNull.Value);cmd.Parameters.AddWithValue("@u",now);cmd.ExecuteNonQuery();return GetState(id);
    }

    public object Filters()
    {
        using var c=Open(); var authors=Strings(c,"SELECT name FROM author ORDER BY name"); if(authors.Count==0)authors=Strings(c,"SELECT DISTINCT author FROM book WHERE author<>'' ORDER BY author");
        var genres=Strings(c,"SELECT name FROM genre ORDER BY name");if(genres.Count==0)genres=Strings(c,"SELECT DISTINCT genre FROM book WHERE genre IS NOT NULL AND genre<>'' ORDER BY genre");
        using var cmd=c.CreateCommand();cmd.CommandText="SELECT MIN(year) FROM book WHERE year IS NOT NULL";var min=cmd.ExecuteScalar();return new{authors,genres,min_year=min is DBNull?null:min,current_year=DateTime.UtcNow.Year};
    }

    public List<string> Authors(string? q,int limit){using var c=Open();using var cmd=c.CreateCommand();cmd.CommandText="SELECT name FROM author WHERE @q='' OR name LIKE @like COLLATE NOCASE ORDER BY name LIMIT @limit";cmd.Parameters.AddWithValue("@q",q??"");cmd.Parameters.AddWithValue("@like",$"%{q}%");cmd.Parameters.AddWithValue("@limit",limit);var list=new List<string>();using var r=cmd.ExecuteReader();while(r.Read())list.Add(r.GetString(0));return list;}

    public (int Added,int Updated,int Skipped,int Removed,List<string> Errors) Reindex(EpubService epub)
    {
        var root=_config().LibraryPath;Directory.CreateDirectory(root);int added=0,updated=0,skipped=0,removed=0;var errors=new List<string>();var seen=new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach(var file in Directory.EnumerateFiles(root,"*.epub",SearchOption.AllDirectories)){try{var full=Path.GetFullPath(file);seen.Add(full);var info=new FileInfo(full);var hash=EpubService.Hash(full);var meta=epub.Metadata(full);using var c=Open();using var check=c.CreateCommand();check.CommandText="SELECT id,sha256 FROM book WHERE path=@p OR sha256=@h LIMIT 1";check.Parameters.AddWithValue("@p",full);check.Parameters.AddWithValue("@h",hash);using var r=check.ExecuteReader();long? id=null;string? oldHash=null;if(r.Read()){id=r.GetInt64(0);oldHash=r.GetString(1);}r.Close();if(id is not null&&oldHash==hash){skipped++;continue;}using var cmd=c.CreateCommand();var now=DateTimeOffset.UtcNow.ToString("O");
            if(id is null){cmd.CommandText="INSERT INTO book(title,author,language,path,size_bytes,modified_iso,sha256,enriched,enrichment_status,genre,year,created_at,updated_at) VALUES(@t,@a,@l,@p,@s,@m,@h,NULL,'none',NULL,NULL,@n,@n)";added++;}else{cmd.CommandText="UPDATE book SET title=@t,author=@a,language=@l,path=@p,size_bytes=@s,modified_iso=@m,sha256=@h,updated_at=@n WHERE id=@id";cmd.Parameters.AddWithValue("@id",id);updated++;}cmd.Parameters.AddWithValue("@t",meta.Title);cmd.Parameters.AddWithValue("@a",meta.Author);cmd.Parameters.AddWithValue("@l",(object?)meta.Language??DBNull.Value);cmd.Parameters.AddWithValue("@p",full);cmd.Parameters.AddWithValue("@s",info.Length);cmd.Parameters.AddWithValue("@m",info.LastWriteTimeUtc.ToString("O"));cmd.Parameters.AddWithValue("@h",hash);cmd.Parameters.AddWithValue("@n",now);cmd.ExecuteNonQuery();EnsureAuthor(c,meta.Author);
        }catch(Exception ex){errors.Add($"{file}: {ex.Message}");}}
        using(var c=Open()){using var cmd=c.CreateCommand();cmd.CommandText="SELECT id,path FROM book";using var r=cmd.ExecuteReader();var gone=new List<long>();while(r.Read()){var p=r.GetString(1);if(!File.Exists(ResolveBookPath(p)))gone.Add(r.GetInt64(0));}r.Close();foreach(var id in gone){using var d=c.CreateCommand();d.CommandText="DELETE FROM book WHERE id=@id;DELETE FROM bookstate WHERE book_id=@id;";d.Parameters.AddWithValue("@id",id);d.ExecuteNonQuery();removed++;}}
        return(added,updated,skipped,removed,errors);
    }

    public string ResolveBookPath(string stored)=>Path.IsPathRooted(stored)?stored:Path.GetFullPath(Path.Combine(_config().LibraryPath,stored));
    public void SetEnrichment(long id,string json,string status,string? genre,int? year){using var c=Open();using var cmd=c.CreateCommand();cmd.CommandText="UPDATE book SET enriched=@e,enrichment_status=@s,genre=COALESCE(@g,genre),year=COALESCE(@y,year),updated_at=@u WHERE id=@id";cmd.Parameters.AddWithValue("@e",json);cmd.Parameters.AddWithValue("@s",status);cmd.Parameters.AddWithValue("@g",(object?)genre??DBNull.Value);cmd.Parameters.AddWithValue("@y",(object?)year??DBNull.Value);cmd.Parameters.AddWithValue("@u",DateTimeOffset.UtcNow.ToString("O"));cmd.Parameters.AddWithValue("@id",id);cmd.ExecuteNonQuery();}
    public List<long> EnrichmentCandidates(bool onlyPending,bool onlyNever,int? limit){using var c=Open();using var cmd=c.CreateCommand();var w=onlyNever?" WHERE enrichment_status='none'":onlyPending?" WHERE enrichment_status<>'ok'":"";cmd.CommandText=$"SELECT id FROM book{w} ORDER BY id"+(limit>0?" LIMIT @limit":"");if(limit>0)cmd.Parameters.AddWithValue("@limit",limit);var ids=new List<long>();using var r=cmd.ExecuteReader();while(r.Read())ids.Add(r.GetInt64(0));return ids;}

    private static Book ReadBook(SqliteDataReader r)=>new(r.GetInt64(0),r.GetString(1),r.GetString(2),N(r,3),r.GetString(4),r.GetInt64(5),r.GetString(6),r.GetString(7),N(r,8),r.GetString(9),N(r,10),NI(r,11),N(r,12),N(r,13));
    private static string? N(SqliteDataReader r,int i)=>r.IsDBNull(i)?null:r.GetString(i); private static long? NL(SqliteDataReader r,int i)=>r.IsDBNull(i)?null:r.GetInt64(i); private static int? NI(SqliteDataReader r,int i)=>r.IsDBNull(i)?null:r.GetInt32(i);private static double? ND(SqliteDataReader r,int i)=>r.IsDBNull(i)?null:r.GetDouble(i);
    private static void Add(SqliteCommand c,Dictionary<string,object?> values){foreach(var p in values)c.Parameters.AddWithValue("@"+p.Key,p.Value??DBNull.Value);}
    private static List<string> Strings(SqliteConnection c,string sql){using var cmd=c.CreateCommand();cmd.CommandText=sql;var x=new List<string>();using var r=cmd.ExecuteReader();while(r.Read())x.Add(r.GetString(0));return x;}
    private static void EnsureAuthor(SqliteConnection c,string author){using var cmd=c.CreateCommand();cmd.CommandText="INSERT OR IGNORE INTO author(name) VALUES(@n)";cmd.Parameters.AddWithValue("@n",author);cmd.ExecuteNonQuery();}
}
