mkdir -p app
mkdir -p app/detection
mkdir -p app/intelligence
mkdir -p app/storage
mkdir -p models
mkdir -p data
mkdir -p data/events
mkdir -p data/snapshots
mkdir -p tests

touch app/__init__.py
touch app/main.py
touch app/config.py
touch app/detection/__init__.py
touch app/detection/detector.py
touch app/intelligence/__init__.py
touch app/intelligence/zone_detector.py
touch app/storage/__init__.py
touch app/storage/database.py
touch README.md

echo "All necessary directories and files have been created!"