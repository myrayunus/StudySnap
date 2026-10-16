# 📚 StudySnap

An AI study assistant that turns lecture notes (PDF) into summaries, flashcards,
key terms, real-world examples, difficulty tags, and an interactive multiple-choice
quiz — powered by **Amazon Bedrock** (Claude Haiku 4.5), deployed to
**AWS Lambda + API Gateway + S3** via **AWS SAM** and **GitHub Actions**.

## Architecture

```
Browser (S3 static site)
  │  buffered JSON  (fetch → { text })        progressive typewriter render on the client
  ▼
8 Lambdas behind one API Gateway (prod stage)
  • Estimated Study Time  ──►  Bedrock Claude Haiku 4.5
  • Bullet Point Summary  ──►  Bedrock Claude Haiku 4.5
  • Key Terms Glossary    ──►  Bedrock Claude Haiku 4.5
  • Real World Examples   ──►  Bedrock Claude Haiku 4.5
  • Difficulty Level Tag  ──►  Bedrock Claude Haiku 4.5
  • Flashcards            ──►  Bedrock Claude Haiku 4.5   (rendered as click-to-flip cards)
  • Interactive Quiz      ──►  Bedrock Claude Haiku 4.5   (chat: history replayed each turn)
  • Study Chat            ──►  Bedrock Claude Haiku 4.5   (free-form Q&A about the notes)
```

Uploads accept **PDF, Word (doc/docx), text (txt/md/csv/html/json), and images (png/jpg)**.
Images and PDFs go to Claude as native blocks; text files are decoded and inlined; other
types are attempted as document blocks.

> **Note on streaming:** the deploy region (`ap-southeast-5`, Malaysia) does not yet support
> Lambda Function URLs (`AWS::Lambda::Url`) or the Lambda Web Adapter layer, so true
> token-by-token streaming isn't available there. Every Lambda is a standard buffered handler
> behind API Gateway, returning the full result as JSON. The frontend reveals text
> progressively (typewriter/fade-in) so the experience still feels live.

## Project layout

```
frontend/index.html                 Single-page app (progressive render, chat, history, adaptive theme)
backend/<widget>/app.py             One buffered Lambda handler per widget (app.handler)
backend/<widget>/requirements.txt   boto3
infra/template.yaml                 AWS SAM template (Lambdas, API Gateway, IAM role, S3)
.github/workflows/deploy.yml        CI/CD: build, deploy, inject URLs, sync to S3
```

## Pre-deployment checklist

### 1. Bedrock model access (region: `ap-southeast-5`)

The old **Bedrock → Model access** console page is retired — serverless foundation models are
auto-enabled on first invocation, and access is now governed by IAM (which this stack's
`AppBedrockRole` already grants). Just make sure your account can invoke these two models in
`ap-southeast-5`:

- `global.anthropic.claude-haiku-4-5-20251001-v1:0` — Claude Haiku 4.5 (global cross-region inference profile)

> First-time Anthropic use may still require the one-time use-case form (Console → Bedrock →
> Model catalog → Claude Haiku 4.5). The `global.` inference profile can route across regions,
> so the IAM policy allows the inference-profile and foundation-model ARNs with wildcard regions.

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
4. `sed`-injects each URL into `frontend/index.html` (replacing the `URL_*` placeholders)
5. `aws s3 sync frontend/` to the website bucket with `--cache-control no-cache`

When it finishes, the **WebsiteUrl** stack output is printed in the final step — that's your live app.

## How the app works

1. **Upload** a PDF (drag & drop or click) — or paste raw notes text. The theme shifts to suit the subject.
2. Click **Run All** to fire every text widget simultaneously (each reveals progressively).
3. Scroll the panels to review the summary, flashcards, glossary, examples, difficulty tags, and study-time estimate.
4. In the **Interactive Quiz** chat, type `start`, then answer each question with `A`, `B`, `C`, or `D`.
5. After 5 questions you get a score out of 5, then rate your confidence (1–5) for personalised study tips.

Each completed session is saved to **browser history** (localStorage) — click a history chip to restore it.

## Local notes

- All Lambdas are standard buffered handlers (`app.handler`) behind a single API Gateway.
  Each returns `{ "text": "..." }`; the frontend renders the markdown progressively so it
  still feels live.
- The quiz Lambda keeps the lecture notes on the first user turn and replays `history` on later turns.
- File uploads are sent as base64 in `file_data` with `file_mime`; PDFs become an Anthropic
  `document` content block, images become an `image` block, prepended to the user message.
