import uvicorn

if __name__ == "__main__":
    print("Booting FastAPI directly through the virtual environment...")
    uvicorn.run("server:app", host="127.0.0.1", port=5000, reload=True)
