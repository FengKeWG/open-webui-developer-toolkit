# oauth.py

`backend/open_webui/utils/oauth.py` implements the optional OAuth login flow.  The module registers providers, handles the redirect callbacks and manages role/group mapping for authenticated users.

## OAuthManager

`OAuthManager` wraps [Authlib](https://docs.authlib.org/) and exposes a few helper methods.  On instantiation every provider listed in `OAUTH_PROVIDERS` calls its `register` function so the FastAPI app can initiate logins.

```python
class OAuthManager:
    def __init__(self, app):
        self.oauth = OAuth()
        self.app = app
        for _, provider_config in OAUTH_PROVIDERS.items():
            provider_config["register"](self.oauth)
```

### Role selection

`get_user_role(user, user_data)` decides which WebUI role to assign after login.  When `ENABLE_OAUTH_ROLE_MANAGEMENT` is true it extracts the roles claim (nested keys are supported) and matches it against the allowed/admin role lists.

```python
oauth_claim = auth_manager_config.OAUTH_ROLES_CLAIM
oauth_allowed_roles = auth_manager_config.OAUTH_ALLOWED_ROLES
oauth_admin_roles = auth_manager_config.OAUTH_ADMIN_ROLES
claim_data = user_data
for nested_claim in oauth_claim.split("."):
    claim_data = claim_data.get(nested_claim, {})
oauth_roles = claim_data if isinstance(claim_data, list) else []
```

If any of the retrieved roles appear in `OAUTH_ADMIN_ROLES` the user becomes an administrator, otherwise the default role is used.

### Group management

`update_user_groups` synchronises membership based on the groups claim.  It can also create missing groups when `ENABLE_OAUTH_GROUP_CREATION` is enabled.  New groups are created by inserting a `GroupForm` via `Groups.insert_new_group`.

```python
if auth_manager_config.ENABLE_OAUTH_GROUP_CREATION:
    for group_name in user_oauth_groups:
        if group_name not in all_group_names:
            new_group_form = GroupForm(
                name=group_name,
                description=f"Group '{group_name}' created automatically via OAuth.",
                permissions=default_permissions,
                user_ids=[],
            )
            Groups.insert_new_group(creator_id, new_group_form)
```

Existing memberships are updated by comparing the current list with the claim and calling `Groups.update_group_by_id` accordingly.

### Profile pictures

`_process_picture_url` downloads the profile image (optionally using an OAuth access token) and returns a base64 data URL.  It falls back to `/user.png` if anything fails.

```python
async with aiohttp.ClientSession() as session:
    async with session.get(picture_url, **get_kwargs) as resp:
        if resp.ok:
            picture = await resp.read()
            base64_encoded_picture = base64.b64encode(picture).decode("utf-8")
            guessed_mime_type = mimetypes.guess_type(picture_url)[0] or "image/jpeg"
            return f"data:{guessed_mime_type};base64,{base64_encoded_picture}"
```

### Login and callback

`handle_login` starts the OAuth flow.  The provider's `authorize_redirect` method is called with the callback URL.

```python
client = self.get_client(provider)
return await client.authorize_redirect(request, redirect_uri)
```

`handle_callback` processes the returned tokens.  It retrieves the user info, validates the email domain and either updates an existing user or creates a new one.  When signups are enabled it generates a random password and stores the provider specific `sub` value so future logins map to the same account.

```python
token = await client.authorize_access_token(request)
user_data: UserInfo = token.get("userinfo") or await client.userinfo(token=token)
provider_sub = f"{provider}@{sub}"
user = Users.get_user_by_oauth_sub(provider_sub)
if not user and auth_manager_config.OAUTH_MERGE_ACCOUNTS_BY_EMAIL:
    user = Users.get_user_by_email(email)
    if user:
        Users.update_user_oauth_sub_by_id(user.id, provider_sub)
```

A JWT is then created with `create_token` and stored in a secure cookie before redirecting back to the frontend.

```python
jwt_token = create_token(data={"id": user.id},
                         expires_delta=parse_duration(auth_manager_config.JWT_EXPIRES_IN))
response.set_cookie(key="token", value=jwt_token,
                    httponly=True,
                    samesite=WEBUI_AUTH_COOKIE_SAME_SITE,
                    secure=WEBUI_AUTH_COOKIE_SECURE)
return RedirectResponse(url=f"{request.base_url}auth#token={jwt_token}", headers=response.headers)
```

Together these helpers provide a pluggable OAuth authentication layer that integrates with WebUI's role and group system.
