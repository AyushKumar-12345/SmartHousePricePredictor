const ctxCurve = document.getElementById('analyticsChart').getContext('2d');
const gradientCurve = ctxCurve.createLinearGradient(0, 0, 0, 200);
gradientCurve.addColorStop(0, 'rgba(99, 102, 241, 0.25)');
gradientCurve.addColorStop(1, 'rgba(99, 102, 241, 0.0)');

const analyticsChart = new Chart(ctxCurve, {
    type: 'line',
    data: {
        labels: ['Min', 'Conservative', 'Target', 'Premium', 'Max'],
        datasets: [{
            data: [124380, 165840, 207300, 248760, 290220],
            borderColor: '#6366f1',
            backgroundColor: gradientCurve,
            tension: 0.4,
            fill: true,
            borderWidth: 3,
            pointBackgroundColor: '#8b5cf6',
            pointHoverBackgroundColor: '#fff',
            pointRadius: 4,
            pointHoverRadius: 6
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { grid: { display: false }, ticks: { color: '#64748b', font: { size: 10, weight: '500' } } },
            y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#64748b', font: { size: 10 } } }
        }
    }
});

const ctxBar = document.getElementById('importanceChart').getContext('2d');
const gradientBar = ctxBar.createLinearGradient(0, 0, 0, 200);
gradientBar.addColorStop(0, '#8b5cf6');
gradientBar.addColorStop(1, '#6366f1');

let importanceChart = new Chart(ctxBar, {
    type: 'bar',
    data: {
        labels: [],
        datasets: [{
            data: [],
            backgroundColor: gradientBar,
            borderRadius: 8,
            borderSkipped: false
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { grid: { display: false }, ticks: { color: '#64748b', font: { size: 9, weight: '500' } } },
            y: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#64748b', font: { size: 9 } } }
        }
    }
});

async function loadImportance() {
    try {
        const response = await fetch('/api/importance');
        const data = await response.json();
        importanceChart.data.labels = Object.keys(data);
        importanceChart.data.datasets[0].data = Object.values(data);
        importanceChart.update();
    } catch (err) {
        console.error(err);
    }
}

document.getElementById('predictorForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const submitBtn = e.target.querySelector('button[type="submit"]');
    submitBtn.disabled = true;
    submitBtn.classList.add('opacity-70');
    
    const formData = new FormData(e.target);
    const payload = Object.fromEntries(formData.entries());

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const resData = await response.json();
        
        if (resData.success) {
            const finalPrice = resData.prediction;
            document.getElementById('outputPrice').innerText = Number(finalPrice).toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0});
            
            analyticsChart.data.datasets[0].data = [
                finalPrice * 0.6,
                finalPrice * 0.8,
                finalPrice,
                finalPrice * 1.2,
                finalPrice * 1.4
            ];
            analyticsChart.update();
        }
    } catch (err) {
        console.error(err);
    } finally {
        submitBtn.disabled = false;
        submitBtn.classList.remove('opacity-70');
    }
});

loadImportance();