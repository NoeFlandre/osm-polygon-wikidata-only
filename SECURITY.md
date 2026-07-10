# Security policy

## Supported version

Security fixes target the latest revision of the `main` branch while the
project is in its `0.x` development series.

## Reporting a vulnerability

Please use the repository's GitHub **Security → Report a vulnerability** flow
to report security issues privately. Do not open a public issue containing
credentials, tokens, private paths, or exploit details.

For ordinary correctness problems that do not expose sensitive information,
use the public [issue tracker](https://github.com/NoeFlandre/osm-polygon-wikidata-only/issues).

## Wikimedia credentials

Treat `WIKIMEDIA_BOT_PASSWORD` as a secret. Do not commit it, store it in a
checked-in `.env` or shell script, include it in logs, paste it into an issue or
pull request, or share it with maintainers. Use a least-privilege Bot Password;
never provide the main Wikimedia account password or exported browser cookies.

If a Bot Password or authenticated cookie may have been disclosed, immediately
open <https://meta.wikimedia.org/wiki/Special:BotPasswords> and revoke the named
credential. Remove it from the current shell with
`unset WIKIMEDIA_BOT_USERNAME WIKIMEDIA_BOT_PASSWORD`, create a replacement if
needed, and privately report any repository exposure through GitHub Security.
Deleting a leaked value from the latest commit is not sufficient because it can
remain in Git history.

The maintainer, Noé Flandre, will acknowledge a private report when reviewed
and coordinate disclosure after a fix is available.
