using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Xml.Linq;

namespace TheArchive;

public sealed class EpubService
{
    public sealed record Meta(string Title,string Author,string? Language);
    public static string Hash(string path){using var stream=File.OpenRead(path);return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();}

    public Meta Metadata(string path)
    {
        try
        {
            using var zip=ZipFile.OpenRead(path);var opf=FindOpf(zip);if(opf is null)throw new InvalidDataException("OPF no encontrado");
            using var s=opf.Open();var doc=XDocument.Load(s);var metadata=doc.Descendants().FirstOrDefault(x=>x.Name.LocalName=="metadata");
            string? V(string name)=>metadata?.Descendants().FirstOrDefault(x=>x.Name.LocalName==name)?.Value?.Trim();
            var fallback=Path.GetFileNameWithoutExtension(path).Replace('_',' ');var parent=Directory.GetParent(path)?.Name?.Replace('_',' ');
            return new(V("title")??fallback,V("creator")??parent??"Desconocido",V("language"));
        }
        catch { var title=Path.GetFileNameWithoutExtension(path).Replace('_',' ');var author=Directory.GetParent(path)?.Name?.Replace('_',' ')??"Desconocido";return new(title,author,null); }
    }

    public string ExtractHtml(string path,int maxChars=200000)
    {
        using var zip=ZipFile.OpenRead(path);var opf=FindOpf(zip)??throw new InvalidDataException("OPF no encontrado");var opfDir=PosixDir(opf.FullName);
        XDocument doc;using(var s=opf.Open())doc=XDocument.Load(s);
        var manifest=doc.Descendants().Where(x=>x.Name.LocalName=="item").ToDictionary(x=>(string?)x.Attribute("id")??"",x=>(string?)x.Attribute("href")??"");
        var spine=doc.Descendants().Where(x=>x.Name.LocalName=="itemref").Select(x=>(string?)x.Attribute("idref")).Where(x=>x is not null).ToList();var parts=new List<string>();var length=0;
        foreach(var id in spine){if(!manifest.TryGetValue(id!,out var href))continue;Append(zip.GetEntry(Combine(opfDir,Uri.UnescapeDataString(href.Split('#')[0]))));if(length>=maxChars)break;}
        if(parts.Count==0)foreach(var entry in zip.Entries.Where(e=>e.FullName.EndsWith(".xhtml",StringComparison.OrdinalIgnoreCase)||e.FullName.EndsWith(".html",StringComparison.OrdinalIgnoreCase))){Append(entry);if(length>=maxChars)break;}
        return string.Join("\n<hr>\n",parts);

        void Append(ZipArchiveEntry? entry){if(entry is null)return;using var sr=new StreamReader(entry.Open(),Encoding.UTF8,true);var html=sr.ReadToEnd();var body=Regex.Match(html,"<body[^>]*>([\\s\\S]*?)</body>",RegexOptions.IgnoreCase).Groups[1].Value;if(string.IsNullOrWhiteSpace(body))body=html;
            body=Regex.Replace(body,"<(script|style|iframe|object|embed)[^>]*>[\\s\\S]*?</\\1>","",RegexOptions.IgnoreCase);body=Regex.Replace(body,"\\s(on\\w+|style)\\s*=\\s*(['\"]).*?\\2","",RegexOptions.IgnoreCase);body=Regex.Replace(body,"(href|src)\\s*=\\s*(['\"])javascript:.*?\\2","$1=\"#\"",RegexOptions.IgnoreCase);parts.Add(body);length+=body.Length;}
    }

    public (byte[] Data,string ContentType)? Cover(string path)
    {
        try{using var zip=ZipFile.OpenRead(path);var opf=FindOpf(zip);if(opf is null)return null;XDocument doc;using(var s=opf.Open())doc=XDocument.Load(s);var items=doc.Descendants().Where(x=>x.Name.LocalName=="item").ToList();var coverId=doc.Descendants().FirstOrDefault(x=>x.Name.LocalName=="meta"&&(string?)x.Attribute("name")=="cover")?.Attribute("content")?.Value;var item=items.FirstOrDefault(x=>(string?)x.Attribute("id")==coverId)??items.FirstOrDefault(x=>((string?)x.Attribute("properties"))?.Contains("cover-image")==true)??items.FirstOrDefault(x=>((string?)x.Attribute("href"))?.Contains("cover",StringComparison.OrdinalIgnoreCase)==true);var href=(string?)item?.Attribute("href");if(href is null)return null;var entry=zip.GetEntry(Combine(PosixDir(opf.FullName),href));if(entry is null)return null;using var ms=new MemoryStream();using(var es=entry.Open())es.CopyTo(ms);var ext=Path.GetExtension(href).ToLowerInvariant();return(ms.ToArray(),ext switch{".png"=>"image/png",".gif"=>"image/gif",".webp"=>"image/webp",".svg"=>"image/svg+xml",_=>"image/jpeg"});}catch{return null;}
    }

    private static ZipArchiveEntry? FindOpf(ZipArchive zip){var container=zip.GetEntry("META-INF/container.xml");if(container is not null){using var s=container.Open();var d=XDocument.Load(s);var path=d.Descendants().FirstOrDefault(x=>x.Name.LocalName=="rootfile")?.Attribute("full-path")?.Value;if(path is not null)return zip.GetEntry(path);}return zip.Entries.FirstOrDefault(e=>e.FullName.EndsWith(".opf",StringComparison.OrdinalIgnoreCase));}
    private static string Combine(string dir,string name)=>string.IsNullOrEmpty(dir)?name:$"{dir.TrimEnd('/')}/{name.TrimStart('/')}";
    private static string PosixDir(string name){var i=name.LastIndexOf('/');return i<0?"":name[..i];}
}
