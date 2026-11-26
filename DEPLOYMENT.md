# Deploy Source Atlas

The FastAPI server serves both the API and the frontend at `/app`. One container hosts both. Uploaded originals require persistent storage; Neon stores metadata and chunks, Pinecone stores vectors, and Groq generates answers. The local BGE embedding model needs more memory than a tiny web service.

## Recommended free host: Oracle Cloud Always Free VM

Oracle [documents](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm) an Always Free Ampere A1 allocation equivalent to **2 OCPUs and 12 GB RAM**, plus a shared block-volume allowance. Availability varies by region. The `deploy/compose.yaml` stack runs the app and Caddy on one VM, with named volumes for uploaded files and the model cache. Caddy provides HTTPS once a domain points to the VM.

1. Create an Always Free Ubuntu Ampere A1 VM in your Oracle home region. Allocate at least 2 OCPUs and enough RAM for the model. In the Oracle network rules and VM firewall, permit TCP 80 and 443. Point a domain's A record to the VM public IP.
2. Install Docker Engine and the Docker Compose plugin using [Docker's Ubuntu instructions](https://docs.docker.com/engine/install/ubuntu/). Clone this repository onto the VM. Docker is not installed in the current Windows workspace, so the image has not been built or tested here.
3. In the cloned repository, copy `.env.example` to `.env`. Set `NEON_DATABASE_URL`, `NEON_MIGRATION_DATABASE_URL` (direct, non-pooler Neon URL), `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `GROQ_API_KEY`, and `APP_ACCESS_PASSWORD`. Set `APP_ACCESS_USERNAME` as desired. Add `DOMAIN=your-real-domain.example` on its own line. Keep `.env` outside Git.
4. From the repository root, run:

   ```bash
   docker compose --env-file .env -f deploy/compose.yaml up -d --build
   docker compose --env-file .env -f deploy/compose.yaml logs -f app
   ```

   The app container applies Alembic migrations before starting. The first answer or upload downloads the embedding model into the persistent `model-cache` volume, so it may take longer.
5. Visit `https://your-real-domain.example/app`. Your browser asks for the configured username and password. Upload a small PDF, ask an answerable question, inspect citations, restart the stack, and confirm the document remains available. Also test deletion and an unanswerable question.

The password gate protects a shared workspace; it does not provide individual user accounts or separate libraries. Use a strong unique password, HTTPS, and non-sensitive demo documents until multi-user access control exists.

## Temporary Render demo

`render.yaml` defines one free Docker web service for the API and frontend. `Dockerfile.render` installs CPU-only PyTorch and caches the BGE embedding model in the image. Render Free has [512 MB RAM and an ephemeral filesystem](https://render.com/docs/free); the model may still exceed that memory limit. A successful build alone does not prove that the service can answer questions. Uploads in `/tmp/uploads` disappear after a restart or redeploy, while Neon records and Pinecone vectors remain. Use separate demo resources when possible, and reupload or delete stale documents after a restart.

1. Push this repository to GitHub. In Render, select **New > Blueprint** and connect the GitHub repository. Render will read `render.yaml` and create `source-atlas-demo`. See [Render's Blueprint guide](https://render.com/docs/infrastructure-as-code).
2. Enter values for each secret prompted by the Blueprint: `APP_ACCESS_PASSWORD`, `NEON_DATABASE_URL`, `NEON_MIGRATION_DATABASE_URL`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, and `GROQ_API_KEY`. The migration URL must be a direct Neon connection string. Never put these values in GitHub. Set a unique strong password; the default username is `admin`.
3. Deploy the service. The container runs Alembic migrations and then starts FastAPI. Wait for `/api/health` to pass, then open the assigned `https://...onrender.com/` URL. It redirects to `/app`, where the browser asks for the configured username and password.
4. Upload a small, non-sensitive PDF or DOCX. Ask a question supported by text in the file, verify the citation, ask a second question, and try an unanswerable question. Check Render logs if the service restarts or says answer generation failed. Test a redeploy before treating the demo as usable.

The Blueprint disables automatic deploys so a later GitHub push does not unexpectedly restart the demo. Trigger a manual deploy in Render when ready. The shared password does not isolate users' documents. For reliable storage and more memory, use the VM deployment above or move originals to object storage and embeddings to a smaller or hosted model.

## Required environment and checks

The server needs outbound HTTPS to Pinecone and Groq, outbound TLS to Neon, and access to the embedding model download during first use. The Pinecone index must be 384-dimensional with cosine similarity for the configured BGE model. Keep `.env` private and rotate any key that has been exposed. The production app refuses to start without `APP_ACCESS_PASSWORD`.

Deployment is **not complete** until a public HTTPS URL has passed upload, ask, and delete checks. Persistence-after-restart cannot pass on Render Free because the original files are ephemeral.

## Temporary public demo

For a short demonstration, a locally running app can be shared through a restricted tunnel. The local computer and app must remain on, and the URL may change. This is not durable hosting. Do not expose an unauthenticated app to the public internet.
