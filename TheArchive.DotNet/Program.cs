using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text.Json;
using TheArchive;

var root=AppContext.BaseDirectory;
if(!File.Exists(Path.Combine(root,"wwwroot","index.html")))root=Directory.GetCurrentDirectory();
var loaded=AppConfig.Load(root);AppConfig current=loaded.Config;string configPath=loaded.Path;
var publicPort=int.TryParse(Environment.GetEnvironmentVariable("THE_ARCHIVE_PUBLIC_PORT"),out var configuredPublicPort)&&configuredPublicPort is >0 and <=65535?configuredPublicPort:current.Port;
var builder=WebApplication.CreateBuilder(args);builder.WebHost.UseUrls($"http://0.0.0.0:{current.Port}");
builder.Services.AddCors(o=>o.AddDefaultPolicy(p=>p.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod()));builder.Services.AddHttpClient();
builder.Services.AddSingleton<Func<AppConfig>>(()=>current);builder.Services.AddSingleton<ArchiveDb>();builder.Services.AddSingleton<EpubService>();builder.Services.AddSingleton<EnrichmentService>();
var app=builder.Build();app.UseCors();app.UseDefaultFiles();app.UseStaticFiles(new StaticFileOptions{ServeUnknownFileTypes=true});
var db=app.Services.GetRequiredService<ArchiveDb>();db.Initialize();var epub=app.Services.GetRequiredService<EpubService>();var enrich=app.Services.GetRequiredService<EnrichmentService>();var progress=new Dictionary<string,object?>{{"running",false}};var progressLock=new object();

IResult Missing(string message)=>Results.NotFound(new{detail=message});
Book? Find(long id)=>db.GetBook(id);
string? BookPath(Book b){var p=db.ResolveBookPath(b.Path);return File.Exists(p)?p:null;}

app.MapGet("/health",()=>new{status="ok",time=DateTimeOffset.UtcNow});
app.MapGet("/legacy",()=>Results.File(Path.Combine(root,"wwwroot","legacy","index.html"),"text/html; charset=utf-8"));
app.MapGet("/ebook",()=>Results.Redirect("/legacy"));
app.MapGet("/api/server-info",()=>new{lan_ip=LocalIp(),port=publicPort,note="The Archive .NET"});
app.MapGet("/api/config",()=>Results.Json(current,AppConfig.JsonOptions()));
app.MapPost("/api/config",async(HttpRequest request)=>{var patch=await JsonSerializer.DeserializeAsync<Dictionary<string,JsonElement>>(request.Body,AppConfig.JsonOptions())??[];var json=JsonSerializer.SerializeToNode(current,AppConfig.JsonOptions())!.AsObject();foreach(var p in patch)if(json.ContainsKey(p.Key))json[p.Key]=System.Text.Json.Nodes.JsonNode.Parse(p.Value.GetRawText());var next=json.Deserialize<AppConfig>(AppConfig.JsonOptions())??current;next.Save(configPath);current=AppConfig.Load(root).Config;db.Initialize();return Results.Json(current,AppConfig.JsonOptions());});
app.MapPost("/config/reload",()=>{current=AppConfig.Load(root).Config;db.Initialize();return Results.Ok(new{status="ok"});});
app.MapGet("/filters",()=>db.Filters());app.MapGet("/authors",(string? q,int? limit)=>db.Authors(q,Math.Clamp(limit??50,1,100)));
app.MapGet("/books",(string? q,string? autor,string? titulo,string? genre,int? year_from,int? year_to,string? state_filter,string? sort_by,int? page,int? page_size)=>{var p=Math.Max(1,page??1);var ps=Math.Clamp(page_size??current.PageSizeDefault,1,200);var result=db.Search(q,autor,titulo,genre,year_from,year_to,state_filter,sort_by,p,ps);return new{total=result.Total,page=p,page_size=ps,items=result.Items};});
app.MapGet("/books/{id:long}",(long id)=>{var b=Find(id);if(b is null)return Missing("Book not found");var s=db.GetState(id);return Results.Ok(new{id=b.Id,title=b.Title,author=b.Author,language=b.Language,path=b.Path,size_bytes=b.SizeBytes,modified_iso=b.ModifiedIso,sha256=b.Sha256,enriched=JsonData.Parse(b.Enriched),enrichment_status=b.EnrichmentStatus,genre=b.Genre,year=b.Year,created_at=b.CreatedAt,updated_at=b.UpdatedAt,favorite=s.Favorite,read=s.Read,pending=s.Pending,cover_url=$"/books/{id}/cover/thumb",full_cover_url=$"/books/{id}/cover"});});
app.MapGet("/books/{id:long}/state",(long id)=>Find(id)is null?Missing("Book not found"):Results.Ok(StateObject(db.GetState(id))));
app.MapPost("/books/{id:long}/state",(long id,StateRequest req)=>Find(id)is null?Missing("Book not found"):Results.Ok(StateObject(db.SaveState(id,state:req))));
app.MapGet("/books/{id:long}/progress",(long id)=>Find(id)is null?Missing("Book not found"):Results.Ok(ProgressObject(db.GetState(id))));
app.MapPost("/books/{id:long}/progress",(long id,ProgressRequest req)=>Find(id)is null?Missing("Book not found"):Results.Ok(ProgressObject(db.SaveState(id,progress:req))));

app.MapGet("/download/{id:long}",(long id)=>Download(id));app.MapGet("/download/{id:long}.epub",(long id)=>Download(id));app.MapGet("/ebook/download/{id:long}",(long id)=>Results.Redirect($"/ebook/raw/{id}.epub"));
app.MapGet("/ebook/raw/{id:long}.epub",(long id)=>Download(id,true));
app.MapGet("/open/{id:long}",(long id)=>{var b=Find(id);if(b is null)return Missing("Book not found");var p=BookPath(b);if(p is null)return Missing("File not found on disk");try{return Results.Content(epub.ExtractHtml(p),"text/html; charset=utf-8");}catch(Exception ex){return Results.Problem(ex.Message);}});
app.MapGet("/books/{id:long}/cover",(long id)=>Cover(id));app.MapGet("/books/{id:long}/cover/thumb",(long id)=>Cover(id));

app.MapGet("/ebook/search",(string? titulo,string? autor,int? page,int? page_size)=>{if(string.IsNullOrWhiteSpace(titulo)&&string.IsNullOrWhiteSpace(autor))return Results.Ok(new{total=0,page=page??1,page_size=page_size??50,items=Array.Empty<object>()});if((!string.IsNullOrWhiteSpace(titulo)&&titulo.Trim().Length<3)||(!string.IsNullOrWhiteSpace(autor)&&autor.Trim().Length<3))return Results.BadRequest(new{detail="La búsqueda debe tener al menos 3 caracteres"});var p=Math.Max(1,page??1);var ps=Math.Clamp(page_size??50,1,200);var r=db.Search(null,autor,titulo,null,null,null,null,"title",p,ps);var small=r.Items.Select(x=>{var e=JsonSerializer.SerializeToElement(x);return new{id=e.GetProperty("id").GetInt64(),title=e.GetProperty("title").GetString(),author=e.GetProperty("author").GetString()};});return Results.Ok(new{total=r.Total,page=p,page_size=ps,items=small});});
app.MapPost("/reindex",(ReindexRequest req)=>{if(req.Mode=="async"){_=Task.Run(()=>db.Reindex(epub));return Results.Ok(new{status="started"});}var r=db.Reindex(epub);return Results.Ok(new{added=r.Added,updated=r.Updated,skipped=r.Skipped,removed=r.Removed,errors=r.Errors});});
app.MapPost("/migrate/backfill",()=>Results.Ok(new{status="ok",note="Las tablas compatibles se mantienen durante la indexación"}));app.MapPost("/migrate/link-authors",()=>Results.Ok(new{status="ok"}));
app.MapPost("/enrich/{id:long}",async(long id,string? language,CancellationToken ct)=>{var b=Find(id);if(b is null)return Missing("Book not found");var r=await enrich.Enrich(b,language,ct);if(r.Status=="ok")SaveEnrichment(id,r.Json);else db.SetEnrichment(id,"null","failed",null,null);return Results.Ok(new{status=r.Status,enriched=JsonData.Parse(r.Json),source=r.Source,error=r.Error});});
app.MapPost("/enrich/batch",async(EnrichBatchRequest req,CancellationToken ct)=>{var ids=req.Ids?.ToList()??db.EnrichmentCandidates(true,false,current.EnrichmentBatchSize);var updated=new List<long>();var failed=new List<long>();foreach(var id in ids){var b=Find(id);if(b is null){failed.Add(id);continue;}var r=await enrich.Enrich(b,req.Language,ct);if(r.Status=="ok"){SaveEnrichment(id,r.Json);updated.Add(id);}else failed.Add(id);}return new{updated,failed};});
app.MapPost("/enrich/all",(EnrichAllRequest req)=>{lock(progressLock){if((bool)(progress["running"]??false))return Results.Ok(new{status="already_running",progress});var ids=db.EnrichmentCandidates(req.OnlyPending,req.OnlyNever,req.Limit);progress=new(){{"running",true},{"total",ids.Count},{"processed",0},{"updated",0},{"failed",0},{"started_at",DateTimeOffset.UtcNow}};_=Task.Run(async()=>{foreach(var id in ids){var b=Find(id);if(b is null)continue;var r=await enrich.Enrich(b,req.Language);lock(progressLock){progress["processed"]=(int)progress["processed"]!+1;if(r.Status=="ok"){SaveEnrichment(id,r.Json);progress["updated"]=(int)progress["updated"]!+1;}else progress["failed"]=(int)progress["failed"]!+1;}if(req.ThrottleMs>0)await Task.Delay(req.ThrottleMs);}lock(progressLock){progress["running"]=false;progress["finished_at"]=DateTimeOffset.UtcNow;}});return Results.Ok(new{status="started",candidates=ids.Count});}});
app.MapGet("/enrich/status",()=>{lock(progressLock)return Results.Ok(progress);});

if(current.OpenBrowser&&!args.Contains("--no-open")){app.Lifetime.ApplicationStarted.Register(()=>{try{Process.Start(new ProcessStartInfo($"http://localhost:{current.Port}/") { UseShellExecute=true });}catch{}});}
Console.WriteLine($"The Archive .NET: http://localhost:{publicPort}/  |  versión antigua: /legacy");app.Run();

IResult Download(long id,bool range=true){var b=Find(id);if(b is null)return Missing("Book not found");var p=BookPath(b);return p is null?Missing("File not found on disk"):Results.File(p,"application/epub+zip",SafeName(b.Title)+".epub",enableRangeProcessing:range);}
IResult Cover(long id){var b=Find(id);if(b is null)return Missing("Book not found");var p=BookPath(b);if(p is null)return Missing("File not found on disk");var c=epub.Cover(p);return c is null?Results.File(Path.Combine(root,"wwwroot","biblioteca.svg"),"image/svg+xml"):Results.File(c.Value.Data,c.Value.ContentType);}
void SaveEnrichment(long id,string json){string? genre=null;int? year=null;try{using var d=JsonDocument.Parse(json);if(d.RootElement.TryGetProperty("genre",out var g))genre=g.GetString();if(d.RootElement.TryGetProperty("year",out var y)&&y.TryGetInt32(out var n))year=n;}catch{}db.SetEnrichment(id,json,"ok",genre,year);}
object StateObject(BookState s)=>new{book_id=s.BookId,favorite=s.Favorite,read=s.Read,pending=s.Pending,updated_at=s.UpdatedAt};object ProgressObject(BookState s)=>new{book_id=s.BookId,mode=s.LastMode,scroll_top=s.ScrollTop,page=s.Page,percent=s.Percent,updated_at=s.UpdatedAt};
static string SafeName(string value)=>string.Concat(value.Select(c=>Path.GetInvalidFileNameChars().Contains(c)?'_':c));
static string LocalIp(){try{using var s=new Socket(AddressFamily.InterNetwork,SocketType.Dgram,ProtocolType.Udp);s.Connect("8.8.8.8",80);return((IPEndPoint)s.LocalEndPoint!).Address.ToString();}catch{return"127.0.0.1";}}
