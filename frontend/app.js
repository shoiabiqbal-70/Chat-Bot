// The UI talks to the same-origin FastAPI API; the browser sends its session
// cookie automatically with each request.
const $ = (id) => document.getElementById(id);
let documentSelectionTouched = false;
let authMode = 'signin';
// Keep account-only document controls hidden until the session is confirmed.
function showWorkspace(account) {
  $('auth-panel').hidden = true;
  document.querySelector('main').hidden = false;
  document.querySelector('footer').hidden = false;
  $('account-email').textContent = account.email;
  $('account-email').hidden = false;
  $('signout').hidden = false;
  refresh();
}
function showAuth() {
  // Clear the previous account's files, question and evidence on sign-out.
  $('auth-panel').hidden = false;
  document.querySelector('main').hidden = true;
  document.querySelector('footer').hidden = true;
  $('account-email').hidden = true;
  $('signout').hidden = true;
  $('documents').replaceChildren();
  $('ask-document').replaceChildren(new Option('All documents', ''));
  $('question').value = '';
  $('document-total').textContent = '0';
  $('ask-status').textContent = '';
  $('upload-status').textContent = '';
  $('evidence').replaceChildren();
  $('answer').textContent = '';
}
function setAuthMode(mode) {
  // One form serves both sign-in and sign-up, including password autocomplete.
  authMode = mode;
  $('signin-tab').classList.toggle('active', mode === 'signin');
  $('signup-tab').classList.toggle('active', mode === 'signup');
  $('auth-submit').textContent = mode === 'signin' ? 'Sign in' : 'Create account';
  $('auth-password').autocomplete = mode === 'signin' ? 'current-password' : 'new-password';
  $('auth-status').textContent = '';
}
$('signin-tab').addEventListener('click', () => setAuthMode('signin'));
$('signup-tab').addEventListener('click', () => setAuthMode('signup'));
$('auth-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  $('auth-submit').disabled = true;
  $('auth-status').textContent = 'Please wait…';
  try {
    // An authenticated response is enough to open the private workspace.
    const account = await api(`/api/auth/${authMode}`, {method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({email:$('auth-email').value, password:$('auth-password').value})});
    $('auth-password').value = '';
    $('auth-status').textContent = '';
    showWorkspace(account);
  } catch (error) { $('auth-status').textContent = error.message; }
  finally { $('auth-submit').disabled = false; }
});
$('signout').addEventListener('click', async () => {
  // Clear local UI even if the network fails. If the server session remains
  // valid, a page refresh will restore it.
  try { await api('/api/auth/signout', {method:'POST'}); } catch {}
  showAuth();
});
async function api(path, options = {}) {
  // Convert FastAPI validation errors and other HTTP failures into UI messages.
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      detail = Array.isArray(body.detail) ? body.detail.map(item => item.msg).join(' ') : (body.detail || detail);
    } catch {}
    if (response.status === 401 && !path.startsWith('/api/auth/')) showAuth();
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}
async function refresh() {
  // Rebuild the library and selector from the server's current document list.
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
      // With one file, select it so document-specific questions work directly.
      selector.selectedIndex = 1;
    } else if (documentSelectionTouched && [...selector.options].some(option => option.value === selected)) {
      selector.value = selected;
    } else if (selector.options.length > 1) {
      selector.selectedIndex = 1;
    }
    if (!docs.length) { list.innerHTML = '<li class="empty-library">Your library is empty. Upload a document to begin.</li>'; return; }
    for (const doc of docs) {
      // textContent keeps uploaded filenames as text, never executable markup.
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
    // Multipart FormData lets the browser set the required upload boundary.
    $('upload-status').textContent = 'Uploading and indexing…';
    const doc = await api('/api/documents', {method:'POST', body:form});
    $('upload-status').textContent = `Indexed ${doc.chunk_count} chunks from ${doc.filename}.`;
    event.target.reset(); await refresh(); $('ask-document').value = doc.id;
    // After upload, questions target that file until the user changes scope.
    documentSelectionTouched = true;
  } catch (error) { $('upload-status').textContent = error.message; await refresh(); }
  finally { button.disabled = false; }
});
$('ask-form').addEventListener('submit', async (event) => {
  event.preventDefault(); const button = event.submitter; button.disabled = true;
  $('ask-status').textContent = 'Searching…'; $('answer').textContent = '';
  $('chunk-count').textContent = ''; $('evidence').replaceChildren();
  // Remove prior citations before a new answer so evidence cannot be mixed.
  try {
    const data = await api('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:$('question').value,
        document_id:$('ask-document').value || null})});
    $('ask-status').textContent = data.abstained ? 'No supported answer found.' : 'Answer grounded in retrieved passages.';
    $('answer').textContent = data.answer;
    $('chunk-count').textContent = `(${data.chunk_count})`;
    for (const chunk of data.chunks) {
      // Render every retrieved passage; mark only verified cited chunk IDs.
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
// Restore a valid session after a page refresh; otherwise show sign-in.
api('/api/auth/me').then(showWorkspace).catch(showAuth);
