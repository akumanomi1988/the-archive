// Legacy-safe JS for frontend/ebook.html
(function(){
  var apiBase = '';
  function el(id){return document.getElementById(id);} 
  function setMsg(s){ el('msg').textContent = s; }
  function clearResults(){ el('results').innerHTML = ''; }
  function showResults(items){
    clearResults();
    if(!items || !items.length) { setMsg('No se encontraron resultados'); return; }
    setMsg('');
    for(var i=0;i<items.length;i++){
      var it = items[i];
      var li = document.createElement('li');
  var downloadUrl = apiBase + '/ebook/download/' + encodeURIComponent(it.id); // interstitial HTML page -> then to .epub
      // For very simple browsers (e-readers) avoid target/_blank and download attributes.
      li.innerHTML = '<strong>'+escapeHtml(it.title)+'</strong>\n'
        + '<div style="color:#6b7280;margin:.35rem 0">'+escapeHtml(it.author || '')+'</div>\n'
        + '<a class="download" href="'+downloadUrl+'">Descargar</a>';
      el('results').appendChild(li);
    }
  }
  function escapeHtml(s){ return String(s||'').replace(/[&<>"']/g, function(m){ return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"}[m]; }); }
  function xhrGet(url, cb){ var r = new XMLHttpRequest(); r.open('GET', url, true); r.onreadystatechange = function(){ if(r.readyState===4){ if(r.status>=200 && r.status<300){ try{ var data = JSON.parse(r.responseText); cb(null,data); } catch(e){ cb(new Error('JSON parse error')); } } else { cb(new Error('HTTP '+r.status)); } } }; r.send(null); }
  function buildQuery(){ var titulo = el('titulo').value.trim(); var autor = el('autor').value.trim(); var q = []; if(titulo) q.push('titulo='+encodeURIComponent(titulo)); if(autor) q.push('autor='+encodeURIComponent(autor)); q.push('page=1'); q.push('page_size=50'); return q.join('&'); }
  function doSearch(){ var titulo = el('titulo').value.trim(); var autor = el('autor').value.trim(); if((titulo && titulo.length<3) || (autor && autor.length<3)){ setMsg('Escribe al menos 3 letras en título o autor'); return; } setMsg('Buscando...'); xhrGet(apiBase + '/ebook/search?' + buildQuery(), function(err,data){ if(err){ setMsg('Error: '+err.message); clearResults(); return; } if(data && data.items){ showResults(data.items); } else { setMsg('Respuesta inválida'); } }); }
  function init(){ el('btnSearch').addEventListener('click', doSearch); el('btnClear').addEventListener('click', function(){ el('titulo').value=''; el('autor').value=''; clearResults(); setMsg(''); }); }
  // Expose simple init after DOM ready
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
