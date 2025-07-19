
from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/devices')
def get_devices():
    devices = [
        {'id': 1, 'name': 'Device 1', 'lat': 51.505, 'lng': -0.09},
        {'id': 2, 'name': 'Device 2', 'lat': 51.51, 'lng': -0.1},
        {'id': 3, 'name': 'Device 3', 'lat': 51.5, 'lng': -0.12}
    ]
    return jsonify(devices)

@app.route('/api/signals')
def get_signals():
    signals = [
        {
            'id': 1,
            'frequency': 144.5,  # MHz
            'signal_strength': -65,  # dBm
            'lat': 51.507,
            'lng': -0.095,
            'detected_by': [1, 2, 3],  # device IDs that detected this signal
            'timestamp': '2023-12-07T10:30:00Z',
            'signal_type': 'FM'
        },
        {
            'id': 2,
            'frequency': 433.92,  # MHz
            'signal_strength': -72,  # dBm
            'lat': 51.502,
            'lng': -0.105,
            'detected_by': [1, 3],
            'timestamp': '2023-12-07T10:32:15Z',
            'signal_type': 'Digital'
        },
        {
            'id': 3,
            'frequency': 868.3,  # MHz
            'signal_strength': -58,  # dBm
            'lat': 51.513,
            'lng': -0.088,
            'detected_by': [2, 3],
            'timestamp': '2023-12-07T10:35:42Z',
            'signal_type': 'LoRa'
        }
    ]
    return jsonify(signals)

if __name__ == '__main__':
    app.run(debug=True) 