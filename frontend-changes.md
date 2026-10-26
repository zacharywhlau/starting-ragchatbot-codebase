# Frontend Code Quality Changes

## Summary

Added Prettier and ESLint to enforce consistent formatting and catch common errors across all frontend files.

## Files Added

### `package.json`
- New file at project root.
- Declares `prettier` and `eslint` + `@eslint/js` as dev dependencies.
- Adds four npm scripts:
  - `lint` — runs ESLint on `frontend/script.js`
  - `lint:fix` — auto-fixes ESLint issues
  - `format` — reformats all frontend JS/CSS/HTML with Prettier
  - `format:check` — verifies formatting without writing (used in CI)
  - `quality` — runs `format:check` then `lint` (full check in one command)

### `.prettierrc`
- Prettier config: single quotes, semicolons, 4-space tabs, 100-char print width, ES5 trailing commas.

### `eslint.config.js`
- ESLint flat config (ESLint v9 format).
- Targets `frontend/**/*.js` with browser + ES2020 globals.
- Rules: `no-unused-vars` (warn), `eqeqeq` (error), `no-var` (error), `prefer-const` (warn).

## Files Modified

### `frontend/script.js`
- Reformatted by Prettier (consistent quotes, trailing commas, indentation).
- Changed `catch (e)` to `catch` (ES2019 optional binding) to eliminate the unused-variable ESLint warning.

### `frontend/index.html`
- Reformatted by Prettier (consistent attribute quoting and indentation).

### `frontend/style.css`
- Reformatted by Prettier (consistent spacing and rule ordering).

## Usage

```bash
# Check formatting and lint in one command
npm run quality

# Auto-format all frontend files
npm run format

# Fix auto-fixable ESLint issues
npm run lint:fix
```
