# TruckScenes Explorer Dashboard

## Setup

### Automatic (Recommended)

Run the setup script, add permissions if needed:

```bash
cd ..
chmod +x setup.sh
./setup.sh
```

### Manual

Download data:

```bash
cd ..
aws s3 sync --no-sign-request s3://man-truckscenes/release/mini/ data/
```

If the AWS cli is not installed, please install using your preferred method.

Then unzip the two zip files downloaded by the aws command in the data folder.

Creating the virtual environment (Python 3.11 required):

```bash
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install truckscenes-devkit[all]
```

Run it:

```bash
cd dashboard
python run.py
```

Then open http://127.0.0.1:5000.
