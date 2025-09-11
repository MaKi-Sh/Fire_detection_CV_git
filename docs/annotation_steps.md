# 🚀 Running Label Studio Locally and Sharing Globally with ngrok

This guide explains how to start Label Studio using Docker and share it over the internet using ngrok.

---

## Step 1. Pip install the Label Studio

```bash
pip install label-studio
```

---
## Step 2. Start ngrok tunnel
```bash
ngrok http --host-header=rewrite 8080
```
---
## Step 3. Copy ngrok temporary link

Such as [https://8ed434c2a4a1.ngrok-free.app](https://8ed434c2a4a1.ngrok-free.app)
---

## Step 4. Start Label Studio with label-studio start

```bash
LABEL_STUDIO_CSRF_TRUSTED_ORIGINS=https://8ed434c2a4a1.ngrok-free.app label-studio start
```

Now open [http://localhost:8080](http://localhost:8080) in your browser to confirm CVAT is working.
---
## Step 5. Stop CVAT

To stop ngrok and label studio, press **Ctrl + C** in the terminal where it’s running.
