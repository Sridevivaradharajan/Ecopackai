import secrets

# Generate SECRET_KEY
print("SECRET_KEY:", secrets.token_urlsafe(32))

# Generate JWT_SECRET
print("JWT_SECRET:", secrets.token_urlsafe(32))