# External dependencies

This project restores only AIDE. Other physics and native MDO dependencies from the source project are excluded.

| Source | Revision | Local changes |
| --- | --- | --- |
| [WecoAI/aideml](https://github.com/WecoAI/aideml) | `6568c8bc27af51fe90ac83cdc87eb59512e4f527` (v0.2.2) | `aideml-poc.patch`, inherited from PhySimAgent |

Run `make -C ENV repos` from the repository root to restore the fixed upstream revision and apply the patch. The patch contains the inherited Claude CLI backend and AIDE compatibility changes. The upstream checkout and its original license remain in the ignored `aideml/` directory; this repository tracks the revision and patch.

Source project: `RunningStone/PhySimAgent` at `f0f5b784fa092f878e9790107582c4c4501130d6`. See `ENV/inheritance.json` for the selective import inventory.
