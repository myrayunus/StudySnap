# 📚 StudySnap

An AI study assistant that turns lecture notes (PDF) into summaries, flashcards,
key terms, real-world examples, difficulty tags, a cute topic illustration, and an
interactive multiple-choice quiz — powered by **Amazon Bedrock** (Claude Haiku 4.5
for text, Nova Canvas for images), deployed to **AWS Lambda + API Gateway + S3** via
**AWS SAM** and **GitHub Actions**.

## Architecture

```
Browser (S3 static site)
  │  token-by-token streaming  (fetch + ReadableStream)
  ▼
7 streaming Flask Lambdas  ──►  Bedrock Claude Haiku 4.5   (Function URLs, RESPONSE_STREAM)
  • Estimated Study Time      via Lambda Web Adapter layer
  • Bullet Point Summary
  • Key Terms Glossary
  • Real World Examples
  • Difficulty Level Tag
  • Flashcards
  • Interactive Quiz (chat)
  │  buffered JSON  (API Gateway)
  ▼
1 image Lambda  ──►  Bedrock Nova Canvas   (standard handler)
  • Topic Visual
```

## Project layout

```
frontend/index.html                 Single-page app (streaming UI, chat, history, adaptive theme)
backend/<widget>/app.py             One Flask streaming app per text widget
backend/<widget>/run.sh             Startup script (python app.py) — Lambda Web Adapter entrypoint
backend/<widget>/requirements.txt   flask + boto3
backend/topic_visual/app.py         Standard Lambda handler calling Nova Canvas
infra/template.yaml                 AWS SAM template (Lambdas, API, IAM role, S3)
.github/workflows/deploy.yml        CI/CD: build, deploy, inject URLs, sync to S3
```

## Pre-deployment checklist

### 1. Enable Bedrock model access (region: `ap-southeast-5`)

In the AWS Console → **Bedrock → Model access**, enable:

- `global.anthropic.claude-haiku-4-5-20251001-v1:0` — Claude Haiku 4.5 (global cross-region inference profile)
- `amazon.nova-canvas-v1:0` — Nova Canvas (image generation)

> First-time accounts may need to submit the one-time use-case form before access is granted.
> The `global.` inference profile routes requests worldwide (incl. us-east-2 / us-west-2) for
> maximum throughput, which is why the IAM policy uses `*` for the foundation-model region.

### 2. Create an S3 bucket for SAM deployment artifacts

```bash
aws s3 mb s3://your-sam-deploy-bucket --region ap-southeast-5
```

### 3. Add GitHub repository secrets

| Secret | Value |
| --- | --- |
| `AWS_ACCESS_KEY_ID` | IAM user/role access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user/role secret key |
| `SAM_DEPLOY_BUCKET` | The bucket name from step 2 |

The deploy credentials need permission to run CloudFormation, create the IAM role,
Lambdas, API Gateway, and the S3 site bucket.

## Deploy

Push to `main` (or run the **Deploy StudySnap** workflow manually via *Actions → Run workflow*).

The workflow:

1. `sam build --template infra/template.yaml`
2. `sam deploy` to the `study-snap` CloudFormation stack in `ap-southeast-5`
3. Reads the stack outputs (Function URLs + API URL)
4. `sed`-injects each URL into `frontend/index.html` (replacing the `URL_*` / `DIAGRAM_API_URL` placeholders)
5. `aws s3 sync frontend/` to the website bucket with `--cache-control no-cache`

When it finishes, the **WebsiteUrl** stack output is printed in the final step — that's your live app.

## How the app works

1. **Upload** a PDF (drag & drop or click) — or paste raw notes text. The theme shifts to suit the subject.
2. Click **Run All** to fire every text widget simultaneously (each streams token-by-token) plus the topic image.
3. Scroll the panels to review the summary, flashcards, glossary, examples, difficulty tags, and study-time estimate.
4. In the **Interactive Quiz** chat, type `start`, then answer each question with `A`, `B`, `C`, or `D`.
5. After 5 questions you get a score out of 5, then rate your confidence (1–5) for personalised study tips.

Each completed session is saved to **browser history** (localStorage) — click a history chip to restore it.

## Local notes

- Text Lambdas use the **Lambda Web Adapter** layer so a normal Flask app streams over a Function URL.
  The streaming contract is plain text: the frontend reads the body with `response.body.getReader()`.
- The quiz Lambda keeps the lecture notes on the first user turn and replays `history` on later turns.
- File uploads are sent as base64 in `file_data` with `file_mime`; PDFs become an Anthropic
  `document` content block, images become an `image` block, prepended to the user message.
