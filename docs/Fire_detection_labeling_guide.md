# 🔥 Fire Detection Dataset Annotation Guide

**Thermo Fisher Junior Innovators Challenge – Fire Detection Science Fair Project**
*by Maki Shinohara (2025)*

---

## 🧮 1. Setup — Installing Python and Label Studio

### 🖡️ Windows / 💻 macOS / 🐧 Linux

#### Step 1. Install Python 3

* Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
* Download and install the latest version (Python ≥ 3.9)
* During installation, check **"Add Python to PATH"**

#### Step 2. Verify installation

```bash
python --version
```

You should see something like `Python 3.10.12`

#### Step 3. Verify or install pip

```bash
python -m pip --version
```

If pip is missing:

```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

#### Step 4. Install Label Studio

```bash
pip install label-studio
```

---

## 🤓 2. Launching Label Studio

After installation:

```bash
label-studio
```

Then open the URL printed in the terminal (usually `http://localhost:8080`).
If prompted, create an account (email + password). It only stores data locally.

---

## 📂 3. Creating a Labeling Project

1. Click **Create Project**
2. Name it: `Fire Detection Annotation - YourName`
3. Click **Create**

### Add the Labeling Interface

Paste this XML in the **Labeling Setup** section:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Fire" background="red"/>
    <Label value="Smoke" background="gray"/>
    <Label value="Nonfire" background="green"/>
  </RectangleLabels>
  <Header value="If there is no fire or smoke, click 'Nonfire' and submit."/>
</View>
```

Click **Save**.

---

## 📁 4. Importing Images

### Step 1. Plug in your USB

Each USB should contain:

```
images/
annotations/
README.txt
```

### Step 2. Import images

In Label Studio:

1. Go to your project → **Import**
2. Click **Upload Files**
3. Select all files inside the `images/` folder on your USB

---

## 🏷️ 5. Annotating Images

Open the **Labeling** tab.

### Labeling Rules

| Label          | When to use              | How                                  |
| -------------- | ------------------------ | ------------------------------------ |
| 🔥 **Fire**    | Flames visible           | Draw a bounding box around the fire  |
| ☁️ **Smoke**   | Visible smoke (no flame) | Draw a bounding box around the smoke |
| 🏠 **Nonfire** | No fire/smoke            | Select "Nonfire" (no box needed)     |

### Keyboard Shortcuts

| Action  | Shortcut                                      |
| ------- | --------------------------------------------- |
| Fire    | `1`                                           |
| Smoke   | `2`                                           |
| Nonfire | `3`                                           |
| Submit  | `Ctrl+Enter` (Win/Linux) or `Cmd+Enter` (Mac) |

---

## 🚀 6. Exporting Work

After labeling:

1. Go to **Project Dashboard**
2. Click **Export**
3. Choose **JSON**
4. Save as:

   ```
   annotations_<yourname>.json
   ```
5. Move it to your USB `annotations/` folder.

---

## 📤 7. Returning Your USB

1. Ensure your `annotations/` folder contains your JSON file.
2. Safely eject your USB.
3. Return it to Maki.

---

## 🔍 8. Troubleshooting

| Problem                 | Solution                                     |
| ----------------------- | -------------------------------------------- |
| Label Studio won’t open | Try `label-studio start` or restart terminal |
| Port in use             | Run `label-studio --port 8090`               |
| Upload errors           | Upload smaller batches (50–100 images)       |
| Project disappeared     | Restart Label Studio; it auto-saves locally  |

---

## 💡 Tips for High-Quality Labeling

* Double-check before submitting.
* If unsure whether it’s smoke or fog, label **Nonfire**.
* Focus on **accuracy > speed**.
* Zoom/pan for small fires.

---

## ✅ Summary

| Step | Action                                            |
| ---- | ------------------------------------------------- |
| 1    | Install Python + pip                              |
| 2    | Install Label Studio (`pip install label-studio`) |
| 3    | Start Label Studio (`label-studio`)               |
| 4    | Create project & paste labeling XML               |
| 5    | Import images from USB `/images/`                 |
| 6    | Annotate (Fire / Smoke / Nonfire)                 |
| 7    | Export JSON to `/annotations/`                    |
| 8    | Return USB to Maki                                |

---

### 🎉 Thank You!

Each labeled image helps train the fire detection AI system for safer homes and communities.
