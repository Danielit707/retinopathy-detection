# HiSight // Retinal Diagnostics Portal

Welcome to the **HiSight Retinal Diagnostics Portal**, an integrated local frontend workspace designed for automated diabetic retinopathy (DR) staging evaluation. This system serves as a secure diagnostic entry point into a local API execution stack, leveraging an **EfficientNet Ensemble model** to analyze high-resolution fundus photographs and return real-time vector classification probabilities.

---

## 🚀 System Architecture & Features

This single-page workspace handles the complete lifecycle of clinical diagnostic reporting, session management, and local data persistence.

* **Secure Node Gateway:** Built-in cryptographic separation between `SIGN IN` and `REGISTER NODE` states. Utilizes isolated HTTP Basic Authentication tokens generated via secure client-side encryption.
* **Fundus Image Ingestion Engine:** Drag-and-drop or local native file-browsing interceptor supporting high-resolution JPEG/PNG diagnostic scans up to 10MB.
* **Live Classification Matrix:** Real-time UI rendering pipeline mapping classification probability vectors directly into dynamic visual analytics bars.
* **Persistent Historical Ledger:** Auto-refreshing patient scanning history tracks linked directly to the node identity session.

---

## 🛠️ Tech Stack & Dependencies

* **Frontend Architecture:** Native Semantic HTML5 & Vanilla ECMAScript (ES6+).
* **Styling Engine:** Tailwind CSS via CDN pipeline.
* **Security Protocol:** Stateless `Authorization: Basic <token>` headers cached client-side in transient operational contexts.
* **Backend Integration Interface:** Built to interface directly with any standard RESTful gateway (e.g., FastAPI, Uvicorn execution layers).

---

## 📁 Repository Structure

```bash
├── index.html          # Unified clinical dashboard and authentication gateway
├── README.md           # System integration documentation
└── backend/            # (Target Local API Stack Instance)

```

---

## 🔌 API Integration Protocol

The frontend interface is configured to consume a local API stack exposed on the system network loop. It expects the following telemetry contract:

### 1. Authentication Layer

* **Registration Node:** `POST /auth/register`
* *Payload:* `application/json` `{"username": "...", "password": "..."}`


* **Data Verification Layer:** Authenticates transactions against target nodes using local storage hashes.

### 2. Inference Execution Staging

* **Predict Processing:** `POST /predict`
* *Payload:* `multipart/form-data` containing the file payload.
* *Headers:* `Authorization: Basic <base64_string>`
* *Expected Response Signature:*
```json
{
  "diagnosis": "Severe DR",
  "confidence": 92.45,
  "probabilities": {
    "No DR": 1.20,
    "Mild DR": 2.15,
    "Moderate DR": 4.20,
    "Severe DR": 92.45,
    "Proliferative DR": 0.00
  }
}

```





### 3. Historical Synchronization

* **History Logs:** `GET /history`
* *Headers:* `Authorization: Basic <base64_string>`



---

## 🔧 Local Workspace Deployment

1. **Clone the Interface Environment:**
```bash
git clone <repository-url>
cd hisight-diagnostics-portal

```


2. **Serve the Client Node:**
Since the architecture uses clean, modern asynchronous native fetch channels, it can be spun up using any local lightweight HTTP serving layer.
Using Python:
```bash
python3 -m http.server 8080

```


Or using Node.js/NPX:
```bash
npx serve .

```


3. **Verify API Lifecycle Connectivity:**
Ensure your localized EfficientNet backend server block is running natively alongside the application port parameters to clear underlying operational loops.

---

## 📊 Evaluation Metrics Staging

The system targets classification indexing aligned with international clinical standards for Diabetic Retinopathy:

* **Class 0:** No DR
* **Class 1:** Mild DR
* **Class 2:** Moderate DR
* **Class 3:** Severe DR
* **Class 4:** Proliferative DR

---

> **System Notice (2026 Operational Loop):** Local storage variables under `hisight_auth` are programmatically flushed on initialization vectors to guarantee high-fidelity development testing environments and clean node connection pipelines.