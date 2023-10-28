from google.cloud import storage


def get_from_bucket(bucket, source, destination):
    storage_client = storage.Client()
    try:
        bucket = storage_client.bucket(bucket)
        blob = bucket.blob(source)
        blob.download_to_filename(destination)
        print(
            f"file: {destination} 'downloaded from bucket: "
            f"{bucket} location: {source} successfully"
        )
    except Exception as e:
        print(e)


def exists_or_fetch(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        source = f"visiontools/{path.name}"
        get_from_bucket("research-models", source, path)
