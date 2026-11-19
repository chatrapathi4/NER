/**
 * Medical NER — Frontend JavaScript
 *
 * Handles:
 *   1. Sending text to the /predict API endpoint
 *   2. Displaying extracted entities as highlighted text and cards
 *   3. Sample sentence selection
 *   4. Health check for model status display
 */

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

const API_BASE = window.location.origin;

// Sample sentences for quick testing
const SAMPLES = [
    "Take Paracetamol 500 mg twice daily for 5 days after food.",
    "Amoxicillin 250 mg orally every 8 hours for 7 days.",
    "Apply Hydrocortisone cream to the affected area once daily.",
    "Administer Insulin 10 units subcutaneously before meals.",
    "Doxycycline 100 mg capsule orally twice daily for 14 days.",
];


// ──────────────────────────────────────────────────────────────────────────────
// Initialization
// ──────────────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    checkHealth();

    // Allow Ctrl+Enter to submit
    document.getElementById("inputText").addEventListener("keydown", (e) => {
        if (e.ctrlKey && e.key === "Enter") {
            extractEntities();
        }
    });
});


// ──────────────────────────────────────────────────────────────────────────────
// Health Check
// ──────────────────────────────────────────────────────────────────────────────

async function checkHealth() {
    const badge = document.getElementById("modelBadge");
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        badge.textContent = data.model_type.replace("_", " ");
        badge.classList.add("active");
    } catch (error) {
        badge.textContent = "Offline";
    }
}


// ──────────────────────────────────────────────────────────────────────────────
// Entity Extraction
// ──────────────────────────────────────────────────────────────────────────────

async function extractEntities() {
    const inputText = document.getElementById("inputText").value.trim();

    if (!inputText) {
        showError("Please enter some medical text to analyze.");
        return;
    }

    // Show loading, hide results/errors
    showLoading(true);
    hideError();
    hideResults();

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: inputText }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP Error: ${response.status}`);
        }

        const data = await response.json();
        showLoading(false);
        displayResults(data);

    } catch (error) {
        showLoading(false);
        showError(`Error: ${error.message}`);
    }
}


// ──────────────────────────────────────────────────────────────────────────────
// Display Results
// ──────────────────────────────────────────────────────────────────────────────

function displayResults(data) {
    const resultsSection = document.getElementById("results");
    const annotatedText = document.getElementById("annotatedText");
    const entityCards = document.getElementById("entityCards");
    const entityCount = document.getElementById("entityCount");
    const jsonOutput = document.getElementById("jsonOutput");

    const entities = data.entities || [];

    // Update entity count
    entityCount.textContent = `${entities.length} entit${entities.length === 1 ? "y" : "ies"} found`;

    // Build annotated text with highlights
    annotatedText.innerHTML = buildAnnotatedHTML(data.text, entities);

    // Build entity cards
    entityCards.innerHTML = "";
    entities.forEach((entity, index) => {
        const card = document.createElement("div");
        card.className = `entity-card ${entity.label}`;
        card.style.animationDelay = `${index * 0.05}s`;
        card.innerHTML = `
            <div class="card-label">${entity.label}</div>
            <div class="card-text">${escapeHTML(entity.text)}</div>
        `;
        entityCards.appendChild(card);
    });

    // Show JSON
    jsonOutput.textContent = JSON.stringify(data, null, 2);

    // Show results section
    resultsSection.classList.remove("hidden");
}


/**
 * Build HTML with entity spans highlighted inline.
 *
 * Sorts entities by start position and inserts highlight <span> tags.
 * Non-entity text is preserved as-is.
 */
function buildAnnotatedHTML(text, entities) {
    if (!entities.length) {
        return escapeHTML(text);
    }

    // Sort entities by start position
    const sorted = [...entities].sort((a, b) => a.start - b.start);

    let html = "";
    let lastEnd = 0;

    sorted.forEach((entity) => {
        // Add text before this entity
        if (entity.start > lastEnd) {
            html += escapeHTML(text.substring(lastEnd, entity.start));
        }

        // Add highlighted entity
        html += `<span class="entity-highlight ${entity.label}">`;
        html += escapeHTML(entity.text);
        html += `<span class="entity-tag">${entity.label}</span>`;
        html += `</span>`;

        lastEnd = entity.end;
    });

    // Add remaining text after last entity
    if (lastEnd < text.length) {
        html += escapeHTML(text.substring(lastEnd));
    }

    return html;
}


// ──────────────────────────────────────────────────────────────────────────────
// Sample Sentences
// ──────────────────────────────────────────────────────────────────────────────

function useSample(index) {
    const textarea = document.getElementById("inputText");
    textarea.value = SAMPLES[index];
    textarea.focus();

    // Auto-extract after a brief visual feedback
    setTimeout(() => extractEntities(), 150);
}


// ──────────────────────────────────────────────────────────────────────────────
// UI Helpers
// ──────────────────────────────────────────────────────────────────────────────

function clearAll() {
    document.getElementById("inputText").value = "";
    hideResults();
    hideError();
    document.getElementById("inputText").focus();
}

function showLoading(show) {
    const loading = document.getElementById("loading");
    const btn = document.getElementById("extractBtn");

    if (show) {
        loading.classList.remove("hidden");
        btn.disabled = true;
    } else {
        loading.classList.add("hidden");
        btn.disabled = false;
    }
}

function hideResults() {
    document.getElementById("results").classList.add("hidden");
}

function showError(message) {
    const errorDiv = document.getElementById("error");
    errorDiv.textContent = message;
    errorDiv.classList.remove("hidden");
}

function hideError() {
    document.getElementById("error").classList.add("hidden");
}

/**
 * Escape HTML special characters to prevent XSS.
 */
function escapeHTML(str) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}
