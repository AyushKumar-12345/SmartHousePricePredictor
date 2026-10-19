/**
 * SmartHousePricePredictor — Client Controller
 * Simple, friendly, and clean JavaScript for house price predictions.
 */

// Sample locations to test easily
const PRESETS = {
    sf_bay: {
        MedInc: 8.50,
        HouseAge: 32,
        AveRooms: 6.2,
        AveBedrms: 1.05,
        Population: 1200,
        AveOccup: 2.5,
        Latitude: 37.77,
        Longitude: -122.42,
        description: "San Francisco Bay Area (High Income / City Center)"
    },
    la_west: {
        MedInc: 5.90,
        HouseAge: 28,
        AveRooms: 5.6,
        AveBedrms: 1.10,
        Population: 1650,
        AveOccup: 2.8,
        Latitude: 34.05,
        Longitude: -118.25,
        description: "Los Angeles Coastal Area"
    },
    sacramento: {
        MedInc: 3.40,
        HouseAge: 22,
        AveRooms: 5.2,
        AveBedrms: 1.05,
        Population: 1750,
        AveOccup: 3.1,
        Latitude: 38.58,
        Longitude: -121.49,
        description: "Sacramento Valley"
    },
    san_diego: {
        MedInc: 6.20,
        HouseAge: 26,
        AveRooms: 5.8,
        AveBedrms: 1.12,
        Population: 1400,
        AveOccup: 2.7,
        Latitude: 32.72,
        Longitude: -117.16,
        description: "San Diego Area"
    }
};

const FEATURE_DISPLAY_NAMES = {
    MedInc: "Area Income",
    AveOccup: "People per House",
    Latitude: "Latitude",
    Longitude: "Longitude",
    AveRooms: "Total Rooms",
    HouseAge: "House Age",
    AveBedrms: "Bedrooms",
    Population: "Population"
};

let importanceChartInstance = null;

// Initialize simple horizontal bar chart
function initImportanceChart(features, weights) {
    const canvas = document.getElementById('importanceChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (importanceChartInstance) {
        importanceChartInstance.destroy();
    }

    // Sort features by weight descending
    const combined = features.map((feat, idx) => ({
        label: FEATURE_DISPLAY_NAMES[feat] || feat,
        weight: weights[idx]
    })).sort((a, b) => b.weight - a.weight);

    const labels = combined.map(item => item.label);
    const data = combined.map(item => (item.weight * 100).toFixed(1));

    // Simple monochrome bar colors
    const barColors = data.map((_, i) => {
        if (i === 0) return '#0F172A'; // Darkest for top factor
        if (i < 3) return '#334155';
        if (i < 6) return '#64748B';
        return '#94A3B8';
    });

    importanceChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: barColors,
                borderRadius: 4,
                borderSkipped: false,
                barThickness: 14
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#0F172A',
                    titleColor: '#FFFFFF',
                    bodyColor: '#F8FAFC',
                    padding: 10,
                    cornerRadius: 6,
                    callbacks: {
                        label: (context) => ` Importance: ${context.parsed.x}%`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: '#F1F5F9' },
                    ticks: {
                        color: '#64748B',
                        font: { size: 10, family: 'Plus Jakarta Sans' },
                        callback: (v) => `${v}%`
                    },
                    title: {
                        display: true,
                        text: 'Factor Importance (%)',
                        color: '#94A3B8',
                        font: { size: 10 }
                    }
                },
                y: {
                    grid: { display: false },
                    ticks: {
                        color: '#0F172A',
                        font: { size: 11, weight: '500', family: 'Plus Jakarta Sans' }
                    }
                }
            }
        }
    });
}

// Fetch feature importances from server
async function loadImportanceData() {
    try {
        const response = await fetch('/api/importance');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        initImportanceChart(Object.keys(data), Object.values(data));
    } catch (err) {
        console.warn("Using fallback factor weights:", err);
        const fallback = {
            MedInc: 0.4363,
            AveOccup: 0.1299,
            Latitude: 0.1190,
            Longitude: 0.1173,
            AveRooms: 0.0809,
            HouseAge: 0.0667,
            AveBedrms: 0.0268,
            Population: 0.0231
        };
        initImportanceChart(Object.keys(fallback), Object.values(fallback));
    }
}

// Apply location preset into form inputs
function applyPreset(key) {
    const preset = PRESETS[key];
    if (!preset) return;

    const form = document.getElementById('predictorForm');
    if (!form) return;

    for (const [field, value] of Object.entries(preset)) {
        if (field === 'description') continue;
        const input = form.elements[field];
        if (input) input.value = value;
    }

    document.querySelectorAll('.preset-pill').forEach(btn => {
        if (btn.dataset.preset === key) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    const presetNote = document.getElementById('presetNote');
    if (presetNote) {
        presetNote.innerText = preset.description;
    }

    submitPredictionForm();
}

// Display error banner
function showError(message) {
    const errorBox = document.getElementById('errorBanner');
    const errorText = document.getElementById('errorMessage');
    if (errorBox && errorText) {
        errorText.innerText = message;
        errorBox.classList.remove('hidden');
    }
}

// Dismiss error banner
function hideError() {
    const errorBox = document.getElementById('errorBanner');
    if (errorBox) {
        errorBox.classList.add('hidden');
    }
}

// Main prediction submission handler
async function submitPredictionForm() {
    hideError();

    const form = document.getElementById('predictorForm');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');

    if (!form || !submitBtn) return;

    // Collect data
    const formData = new FormData(form);
    const payload = {};
    for (const [key, value] of formData.entries()) {
        payload[key] = parseFloat(value);
    }

    // Client-side quick checks
    if (isNaN(payload.MedInc) || payload.MedInc <= 0) {
        showError("Please enter a valid positive number for Area Income.");
        return;
    }
    if (payload.AveBedrms > payload.AveRooms) {
        showError(`Number of bedrooms (${payload.AveBedrms}) cannot be more than total rooms (${payload.AveRooms}).`);
        return;
    }

    // Show loading state
    submitBtn.disabled = true;
    if (btnText) btnText.innerText = "Predicting Price...";
    if (btnSpinner) btnSpinner.classList.remove('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Could not calculate price. Please check your inputs.");
        }

        // Render predictions
        const priceNum = data.prediction;
        const formattedPrice = data.formatted_prediction || `$${Number(priceNum).toLocaleString()}`;

        const priceEl = document.getElementById('outputPrice');
        if (priceEl) priceEl.innerText = formattedPrice;

        if (data.range) {
            const lowerEl = document.getElementById('lowerRange');
            const upperEl = document.getElementById('upperRange');
            if (lowerEl) lowerEl.innerText = data.range.lower_formatted;
            if (upperEl) upperEl.innerText = data.range.upper_formatted;
        }

        // California 1990 Average Benchmark: $206,855
        const censusMedian = 206855.0;
        const diffRatio = (priceNum - censusMedian) / censusMedian;
        const diffPct = Math.abs(diffRatio * 100).toFixed(0);

        const compEl = document.getElementById('comparisonBadge');
        if (compEl) {
            if (diffRatio >= 0) {
                compEl.innerText = `+${diffPct}% above California average ($206,855)`;
                compEl.className = "text-xs font-semibold text-emerald-800 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-full";
            } else {
                compEl.innerText = `-${diffPct}% below California average ($206,855)`;
                compEl.className = "text-xs font-semibold text-slate-700 bg-slate-100 border border-slate-200 px-2.5 py-1 rounded-full";
            }
        }

        // Progress meter
        const meter = document.getElementById('valuationMeter');
        if (meter) {
            const clampedPct = Math.min(100, Math.max(5, ((priceNum - 50000) / (500000 - 50000)) * 100));
            meter.style.width = `${clampedPct}%`;
        }

    } catch (err) {
        console.error("Prediction error:", err);
        showError(err.message || "Failed to calculate valuation. Please verify your inputs.");
    } finally {
        submitBtn.disabled = false;
        if (btnText) btnText.innerText = "Predict House Price";
        if (btnSpinner) btnSpinner.classList.add('hidden');
    }
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictorForm');
    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            document.querySelectorAll('.preset-pill').forEach(btn => btn.classList.remove('active'));
            submitPredictionForm();
        });
    }

    // Attach preset button clicks
    document.querySelectorAll('.preset-pill').forEach(btn => {
        btn.addEventListener('click', () => {
            const presetKey = btn.dataset.preset;
            if (presetKey) applyPreset(presetKey);
        });
    });

    const dismissBtn = document.getElementById('dismissErrorBtn');
    if (dismissBtn) {
        dismissBtn.addEventListener('click', hideError);
    }

    loadImportanceData();
    submitPredictionForm();
});