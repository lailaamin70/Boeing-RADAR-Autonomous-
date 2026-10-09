from app import create_app

app = create_app()

if __name__ == "__main__":
    # debug should be turned off later
    app.run(debug=True, port=5000)
