# Docker image build

The Dockerfile uses a multi-stage build to keep compiler tools out of the final runtime image. The service runs as a non-root user and exposes a health check. Dependencies are installed from a pinned lock file so repeated builds resolve the same versions.
