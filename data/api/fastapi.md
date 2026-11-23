# FastAPI dependency injection

FastAPI resolves dependencies declared with Depends before calling an endpoint function. A shared dependency can validate a request header, create a database session, or expose application settings. For a database session, a dependency that yields the session can close it after the request finishes. Dependency overrides let tests replace the database session with a test session without changing the route implementation.

Use an async endpoint when its downstream libraries provide asynchronous I/O. A synchronous database driver called inside an async endpoint can block the event loop. The application should make the lifetime of a session explicit and avoid sharing one mutable session between concurrent requests.
