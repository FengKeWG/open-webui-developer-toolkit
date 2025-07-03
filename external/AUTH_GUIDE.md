# auth.py

`backend/open_webui/utils/auth.py` centralizes authentication helpers for the Open WebUI server. It covers password hashing, JWT creation, API key handling and license checks.  The module is imported by the REST routes and middleware whenever user info needs to be verified.

## Signature verification

`verify_signature(payload, signature)` checks an HMAC generated with the `TRUSTED_SIGNATURE_KEY` environment value.  It is used by the license endpoint to validate incoming data:

```python
expected = base64.b64encode(
    hmac.new(TRUSTED_SIGNATURE_KEY, payload.encode(), hashlib.sha256).digest()
).decode()

return hmac.compare_digest(expected, signature)
```

## Overriding static assets via licenses

`get_license_data(app, key)` contacts `https://api.openwebui.com/api/v1/license/` and applies any returned resources or metadata.  Files are stored under `STATIC_DIR` using `override_static` which decodes the base64 content and writes it to disk.

```python
res = requests.post(
    "https://api.openwebui.com/api/v1/license/",
    json={"key": key, "version": "1"},
    timeout=5,
)
if res.ok:
    payload = res.json()
    for path, content in payload.get("resources", {}).items():
        override_static(path, content)
```

This mechanism lets a license bundle custom logos or HTML fragments.

## Password helpers

The module configures `passlib` with bcrypt via `CryptContext` and exposes two simple wrappers:

- `verify_password(plain, hashed)`
- `get_password_hash(password)`

```python
hashed = get_password_hash("secret")
assert verify_password("secret", hashed)
```

## Token utilities

JWTs are signed with `WEBUI_SECRET_KEY` using HS256. `create_token` accepts a payload plus an optional expiry and returns the encoded string. `decode_token` verifies the signature and returns the decoded dict or `None`.

```python
payload = {"id": user.id}
access = create_token(payload, timedelta(days=1))
claims = decode_token(access)
```

`create_api_key` generates opaque identifiers prefixed with `"sk-"` for programmatic access.

## Extracting the current user

`get_current_user(request, background_tasks, auth_token)` resolves who is making a request.  The token can be supplied via the `Authorization` header, a cookie or as an API key.  When an API key is used the route must opt‑in by setting `request.state.enable_api_key`.

Pseudo code overview:

```python
if token.startswith("sk-"):
    if not request.state.enable_api_key:
        raise HTTPException(403)
    return get_current_user_by_api_key(token)

data = decode_token(token)
user = Users.get_user_by_id(data["id"])
background_tasks.add_task(Users.update_user_last_active_by_id, user.id)
return user
```

`get_current_user_by_api_key` looks up the key in the database and updates the user's `last_active_at` timestamp.  Higher level dependencies `get_verified_user` and `get_admin_user` restrict access based on the `role` field.

## HTTP helpers

`extract_token_from_auth_header` and `get_http_authorization_cred` parse the `Authorization` header when FastAPI's `HTTPBearer` class is not used directly.

```python
cred = get_http_authorization_cred("Bearer " + token)
current = get_current_user(request, tasks, cred)
```

Together these functions provide the authentication layer consumed by the other utilities documented in this folder.
