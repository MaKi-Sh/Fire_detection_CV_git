# Label Studio Annotation Setup — macOS & Windows (Step‑by‑Step)

This guide standardizes how to set up Label Studio, connect local USB folders as data sources/targets, use a consistent labeling interface, and export annotations.

---

## 0) What you need

* **Python 3.9+** installed
* **Label Studio** (`pip install label-studio`)
* A USB with:

  * `Raw_files/` (images to annotate)
  * `Annotated_files/` (empty; where JSON exports will go)

---

## 1) Install & launch Label Studio

### macOS

```bash
# install once
pip install label-studio

# start LS in the same terminal where you set env vars (see §2)
label-studio start
```

### Windows (PowerShell)

```powershell
# install once
python -m pip install label-studio

# start LS in the same PowerShell where you set env vars (see §2)
label-studio start
```

Open the URL shown in the terminal (usually `http://localhost:8080`). Create a local account if prompted.

---

## 2) Enable Local File Storage (so we can use USB folders directly)

> You must set these environment variables **before** starting Label Studio, and start LS from the **same shell**.

### macOS (Terminal / zsh)

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="/Volumes"
label-studio start
```

* `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` must be a parent folder of your data. For USBs on macOS, `/Volumes` is correct (ex: `/Volumes/USB_1/Raw_files`).

### Windows (PowerShell)

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "C:\\"
label-studio start
```

* Set the root to a safe parent folder that contains your data (e.g., `D:\` if your USB is `D:`). You can also scope it narrowly, e.g., `C:\Users\YourName\FireData`.

> **Tip:** To make these permanent, add them to your shell profile (`~/.zshrc` on macOS, or your PowerShell profile) and always launch LS from that shell.

---

## 3) Create Project & Add Source/Target Storage

### 3.1 Create a project

1. Open LS → **Create Project** → Name it: `Fire Detection Annotation - <YourName>`
2. Click **Create**

### 3.2 Add **Source Storage** (Local Files)

* Go to **Project → Settings → Cloud Storage → Add Source Storage**
* **Storage type:** Local Files
* **Name:** `USB_1_Raw`
* **Absolute local path (macOS):** `/Volumes/USB_1/Raw_files`
* **Absolute local path (Windows):** `D:\Raw_files` (replace `D:` with your USB letter)
* **File Filter Regex:** `.*`
* **Import method:** **Files** 
* **Recursive:** (recommended)
* Click **Save and Sync** → tasks appear in Data Manager

### 3.3 Add **Target Storage** (for exports)

* **Add Target Storage** → **Local Files**
* **Name:** `USB_1_Annotations`
* **Absolute local path (macOS):** `/Volumes/USB_1/Annotated_files`
* **Absolute local path (Windows):** `D:\Annotated_files`
* **Format:** JSON
* Click **Add Storage** (you'll use **Export → Sync to Target Storage** later)

> You can repeat 3.2–3.3 for multiple USBs: `USB_2_Raw` → `/Volumes/USB_2/Raw_files`, etc.

---

## 4) Standard Labeling Interface (paste this XML)

Go to **Project → Settings → Labeling Interface → Code View** and paste:

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image" showInline="true">
    <Label value="Fire" background="red" hotkey="1"/>
    <Label value="Smoke" background="gray" hotkey="2"/>
    <Label value="Nonfire" background="green" hotkey="3"/>
  </RectangleLabels>
  <Choices name="scene_choice" toName="image" choice="single" showInline="true">
    <Choice value="Entire Image Nonfire" hotkey="n"/>
  </Choices>
</View>
```

Click **Save**.

**Policy:**

* **Fire** → box around visible flames
* **Smoke** → box around visible smoke/haze
* **Nonfire** → box around *false positives* (glare, sun, red lights, reflections, fog)
* **Entire Image Nonfire** → choose when *no fire/smoke anywhere* (no boxes required)

Hotkeys: `1` Fire, `2` Smoke, `3` Nonfire, `N` Entire Image Nonfire, **Ctrl/Cmd + Enter** Submit

---

## 5) Labeling Workflow

1. Open **Labeling** tab → annotate each task per the policy
2. Don’t skip tasks; either draw boxes or check **Entire Image Nonfire**
3. Submit with **Ctrl/Cmd + Enter**

> You can pause anytime. If using USB Source Storage, re‑mounting the USB at the **same path** (macOS: `/Volumes/USB_1`, Windows: same drive letter) will restore previews. Annotations are saved in LS’s local database.

---

## 6) Exporting Annotations

### Option A — Sync to Target Storage (recommended)

* Go to **Export** → **Sync to Target Storage** → choose `USB_1_Annotations`
* A JSON file like `project-<id>-annotations.json` is written to the USB `Annotated_files/`

### Option B — Manual download

* **Export** → choose **JSON** → download
* Copy the JSON into `Annotated_files/` on the USB

---

## 7) Optional: CLI import (organizer fast path)

Avoids browser upload limits; registers files directly.

Find your project ID from URL (e.g., `/projects/2/` → ID `2`). Then:

### macOS

```bash
label-studio import \
  "/Users/<you>/Library/Application Support/label-studio/projects/2" \
  "/Volumes/USB_1/Raw_files"
```

### Windows (PowerShell)

```powershell
label-studio import \
  "C:\\Users\\<you>\\AppData\\Local\\label-studio\\projects\\2" \
  "D:\\Raw_files"
```

> Replace `<you>`, `2`, and drive letters accordingly.

---

## 8) Troubleshooting

* **Validation error: 'name' is a required property** → Ensure interface XML uses `name` on controls and correct `toName` links (use the XML in §4).
* **UI upload error: “You cannot access body after reading from request's data stream”** → Use **Source Storage** (this guide) or import in smaller batches, or use CLI import in §7.
* **Images not showing after re‑plugging USB** → Confirm the mount path is identical (macOS `/Volumes/USB_1` must not become `/Volumes/USB_1 1`; on Windows, the drive letter must match).
* **Permissions** → If LS can’t read/write to the USB path, check OS permissions and that your **Document Root** (env var) is a parent of the specified absolute paths.

---

## 9) Quality Checklist for Annotators

* 🔲 Label **all** flames and smoke
* 🔲 Mark **false positives** (glare/lights/fog) with **Nonfire** boxes
* 🔲 Use **Entire Image Nonfire** when applicable
* 🔲 Keep boxes tight; avoid huge sloppy boxes
* 🔲 Submit each task (Ctrl/Cmd + Enter)
* 🔲 Export/Sync JSON into `Annotated_files/` before returning USB

---

## 10) Quick Reference (mount paths)

* **macOS USB:** `/Volumes/USB_#/Raw_files` and `/Volumes/USB_#/Annotated_files`
* **Windows USB:** `D:\Raw_files` and `D:\Annotated_files` (replace `D:` with actual drive letter)

---

**You’re ready to annotate. Thank you!** Each labeled image directly improves our fire detection model’s reliability and lowers false positives.
---

## 💾 USB Safety — How to Stop Label Studio Properly

Always **shut down Label Studio before unplugging the USB**  
to avoid data loss or broken file paths.

### 🧱 Why
Label Studio keeps image files open from the USB (e.g., `/Volumes/USB_1/Raw_files`).  
If you remove it while LS is running, annotations might not save correctly or  
the project could lose links to your data.

---

### 🧩 macOS / Linux
1. In the Terminal where Label Studio is running, press:
   ```bash
   Ctrl + C

