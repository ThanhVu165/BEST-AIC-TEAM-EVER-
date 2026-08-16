# UI — Person 3

Own this module. Use Streamlit.

## Modes

### Development/debug

Show query, task, ranked candidates, frame/video preview and component scores exposed by the API.

### Competition/operator

Load/enter query, run search, inspect returned candidates, and trigger submission export.

## Rules

- UI talks to FastAPI only.
- UI does not import Query Engine internals.
- UI does not edit, reorder or overwrite model predictions.
- UI must not depend on physical paths for videos/frames; use API endpoints.
- Use mock API responses during independent development.
