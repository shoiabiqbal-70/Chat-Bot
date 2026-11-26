const $ = (id) => document.getElementById(id);
let documentSelectionTouched = false;
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try { detail = (await response.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}
async function refresh() {
  const list = $('documents');
  const selector = $('ask-document');
  const selected = selector.value;
  list.replaceChildren();
  try {
    const docs = await api('/api/documents');
    $('document-total').textContent = String(docs.length);
    selector.replaceChildren(new Option('All documents', ''));
    for (const doc of docs) {
      if (doc.status === 'completed') selector.add(new Option(doc.filename, doc.id));
    }
    if (selector.options.length === 2) {
      selector.selectedIndex = 1;
    } else if (documentSelectionTouched && [...selector.options].some(option => option.value === selected)) {
      selector.value = selected;
    } else if (selector.options.length > 1) {
      selector.selectedIndex = 1;
    }
    if (!docs.length) { list.innerHTML = '<li class="empty-library">Your library is empty. Upload a document to begin.</li>'; return; }
    for (const doc of docs) {
      const item = document.createElement('li');
      const head = document.createElement('div'); head.className = 'doc-head';
      const name = document.createElement('span'); name.className = 'filename'; name.textContent = doc.filename;
      const reindex = document.createElement('button'); reindex.textContent = 'Reindex'; reindex.type = 'button';
      reindex.addEventListener('click', async () => {
        reindex.disabled = true; $('upload-status').textContent = `Reindexing ${doc.filename}…`;
        try {
          const result = await api(`/api/documents/${doc.id}/reindex`, {method:'POST'});
          $('upload-status').textContent = `Reindexed ${result.chunk_count} chunks from ${doc.filename}.`;
          await refresh();
        } catch (error) { $('upload-status').textContent = error.message; reindex.disabled = false; }
      });
      const remove = document.createElement('button'); remove.textContent = 'Delete'; remove.type = 'button';
      remove.addEventListener('click', async () => {
        remove.disabled = true;
        try { await api(`/api/documents/${doc.id}`, {method:'DELETE'}); await refresh(); }
        catch (error) { $('upload-status').textContent = error.message; remove.disabled = false; }
      });
      head.append(name, reindex, remove);
      const meta = document.createElement('p'); meta.className = 'meta'; meta.textContent = `${doc.status} · ${new Date(doc.created_at).toLocaleString()}`;
      item.append(head, meta); list.append(item);
    }
  } catch (error) { list.textContent = error.message; $('document-total').textContent = '—'; }
}
$('refresh').addEventListener('click', refresh);
$('ask-document').addEventListener('change', () => { documentSelectionTouched = true; });
$('upload-form').addEventListener('submit', async (event) => {
  event.preventDefault(); const button = event.submitter; button.disabled = true;
  $('upload-status').textContent = 'Uploading…';
  try {
    const form = new FormData(event.target);
    $('upload-status').textContent = 'Uploading and indexing…';
    const doc = await api('/api/documents', {method:'POST', body:form});
    $('upload-status').textContent = `Indexed ${doc.chunk_count} chunks from ${doc.filename}.`;
    event.target.reset(); await refresh(); $('ask-document').value = doc.id;
    documentSelectionTouched = true;
  } catch (error) { $('upload-status').textContent = error.message; await refresh(); }
  finally { button.disabled = false; }
});
$('ask-form').addEventListener('submit', async (event) => {
  event.preventDefault(); const button = event.submitter; button.disabled = true;
  $('ask-status').textContent = 'Searching…'; $('answer').textContent = '';
  $('chunk-count').textContent = ''; $('evidence').replaceChildren();
  try {
    const data = await api('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:$('question').value,
        document_id:$('ask-document').value || null})});
    $('ask-status').textContent = data.abstained ? 'No supported answer found.' : 'Answer grounded in retrieved passages.';
    $('answer').textContent = data.answer;
    $('chunk-count').textContent = `(${data.chunk_count})`;
    for (const chunk of data.chunks) {
      const node = document.createElement('div'); node.className = 'source';
      const title = document.createElement('p'); title.className = 'meta';
      title.textContent = `${data.citations.includes(chunk.chunk_id) ? 'Cited · ' : ''}${chunk.filename} · page ${chunk.page_number ?? '—'} · chunk ${chunk.chunk_index}`;
      const score = document.createElement('p'); score.className = 'score'; score.textContent =
        chunk.similarity_score === 0 ? 'Document passage' : `Cosine similarity: ${chunk.similarity_score.toFixed(3)}`;
      const body = document.createElement('p'); body.textContent = chunk.text;
      node.append(title, score, body); $('evidence').append(node);
    }
  } catch (error) { $('ask-status').textContent = error.message; }
  finally { button.disabled = false; }
});
refresh();
