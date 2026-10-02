# Security policy

## Supported version

The project is in the `0.x` development series. Security fixes apply to the latest revision of the `main` branch.

## Reporting a vulnerability

Use the GitHub flow **Security → Report a vulnerability** to report a security issue in private. Do not open a public issue that contains credentials, tokens, private paths, or exploit details.

For an ordinary correctness problem that does not show sensitive information, use the public [issue tracker](https://github.com/NoeFlandre/osm-polygon-wikidata-only/issues).

## Wikimedia credentials

Treat `WIKIMEDIA_BOT_PASSWORD` as a secret. Do not do any of these actions:

- Commit it.
- Store it in a `.env` file or a shell script that is in the repository.
- Put it in logs.
- Paste it into an issue or a pull request.
- Share it with the maintainers.

Use a Bot Password with the fewest privileges. Never give the main Wikimedia account password or exported browser cookies.

If a Bot Password or an authenticated cookie is possibly exposed, do these steps:

1. Open <https://meta.wikimedia.org/wiki/Special:BotPasswords> immediately.
2. Revoke the named credential.
3. Remove it from the current shell with `unset WIKIMEDIA_BOT_USERNAME WIKIMEDIA_BOT_PASSWORD`.
4. Create a new credential if necessary.
5. Report the repository exposure in private through GitHub Security.

Do not only delete the leaked value from the latest commit. The value stays in the Git history.

The maintainer, Noé Flandre, acknowledges a private report when he reviews it. He then coordinates the disclosure after a fix is available.
