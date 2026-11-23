# JWT verification

The API verifies the JWT signature, issuer, audience, and expiration before accepting a protected request. The signing key is loaded from a secret manager, not committed to the repository. Authorization checks use claims only after signature validation.
