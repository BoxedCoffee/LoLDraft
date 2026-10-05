// Global variable to store champions list
let champions = [];

// Load champion list when page loads
document.addEventListener('DOMContentLoaded', function() {
    fetch('/champions')
        .then(response => response.json())
        .then(data => {
            champions = data;
            populateChampionSelects();
        })
        .catch(error => {
            console.error('Error loading champions:', error);
            // Fallback: use some sample champions
            champions = [
                {"id": "1", "name": "Annie", "title": "the Dark Child", "image": "https://ddragon.leagueoflegends.com/cdn/14.20.1/img/champion/Annie.png"},
                {"id": "2", "name": "Olaf", "title": "the Berserker", "image": "https://ddragon.leagueoflegends.com/cdn/14.20.1/img/champion/Olaf.png"}
            ];
            populateChampionSelects();
        });
});

// Populate champion selects for both teams
function populateChampionSelects() {
    const blueContainer = document.getElementById('blue-champions');
    const redContainer = document.getElementById('red-champions');

    // Clear existing inputs
    blueContainer.innerHTML = '';
    redContainer.innerHTML = '';

    // Create 5 inputs for each team
    for (let i = 0; i < 5; i++) {
        createChampionInput(blueContainer, 'blue', i);
        createChampionInput(redContainer, 'red', i);
    }
}

// Create a single champion input field
function createChampionInput(container, team, index) {
    const div = document.createElement('div');
    div.className = 'champion-input';

    const label = document.createElement('label');
    label.textContent = `${team === 'blue' ? 'Blue' : 'Red'} Team - ${getRoleName(index)}`;
    label.setAttribute('for', `${team}-champ-${index}`);

    const select = document.createElement('select');
    select.id = `${team}-champ-${index}`;
    select.name = `${team}_champ_${index}`;

    // Add default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Select Champion';
    select.appendChild(defaultOption);

    // Add champion options with images
    champions.forEach(champion => {
        const option = document.createElement('option');
        option.value = champion.id;
        option.textContent = champion.name;
        // Add image data to option for display
        option.dataset.image = champion.image;
        select.appendChild(option);
    });

    div.appendChild(label);
    div.appendChild(select);
    container.appendChild(div);
}

// Helper function to get role names
function getRoleName(index) {
    const roles = ['Top', 'Jungle', 'Mid', 'Bot', 'Support'];
    return roles[index];
}

// Make prediction when button is clicked
document.getElementById('predict-btn').addEventListener('click', function() {
    // Collect champion data for both teams
    const blueTeam = [];
    const redTeam = [];

    for (let i = 0; i < 5; i++) {
        const blueChamp = document.getElementById(`blue-champ-${i}`).value;
        if (blueChamp) blueTeam.push(blueChamp);

        const redChamp = document.getElementById(`red-champ-${i}`).value;
        if (redChamp) redTeam.push(redChamp);
    }

    // Check if we have champions for both teams
    if (blueTeam.length === 0 || redTeam.length === 0) {
        alert('Please select at least one champion for each team');
        return;
    }

    // Make prediction request
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            blue_team: blueTeam,
            red_team: redTeam
        })
    })
    .then(response => response.json())
    .then(data => {
        displayResults(data);
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('results').innerHTML = '<p style="color: red;">Error making prediction. Please try again.</p>';
    });
});

// Display results
function displayResults(prediction) {
    const resultsDiv = document.getElementById('results');

    if (prediction.error) {
        resultsDiv.innerHTML = `<p style="color: red;">Error: ${prediction.error}</p>`;
        return;
    }

    // Create HTML for results
    let html = `
        <div class="result-section">
            <h3>Win Probability</h3>
            <p><strong>${(prediction.win_probability * 100).toFixed(1)}%</strong></p>
        </div>

        <div class="result-section">
            <h3>Gold Curve Prediction</h3>
            <div class="gold-curve">
    `;

    // Add gold curve points
    const goldPoints = ['5 min', '10 min', '15 min', '20 min', '25 min', '30 min'];
    prediction.gold_curve.forEach((value, index) => {
        html += `
            <div class="gold-point">
                <div>${goldPoints[index]}</div>
                <div><strong>${value > 0 ? '+' : ''}${value}</strong></div>
            </div>
        `;
    });

    html += `
            </div>
        </div>

        <div class="result-section">
            <h3>Objective Predictions</h3>
            <p>Each objective type (Dragon, Herald, Baron, Tower):</p>
    `;

    // Add objective predictions
    const objTypes = ['Dragon', 'Herald', 'Baron', 'Tower'];
    prediction.objective_predictions.forEach((value, index) => {
        let text = '';
        let className = 'objective-none';

        switch(value) {
            case 0:
                text = 'None';
                break;
            case 1:
                text = 'Blue Team';
                className = 'objective-blue';
                break;
            case 2:
                text = 'Red Team';
                className = 'objective-red';
                break;
        }

        html += `
            <div class="objective-prediction ${className}">
                ${objTypes[index]}: ${text}
            </div>
        `;
    });

    html += `
        </div>
    `;

    resultsDiv.innerHTML = html;
}