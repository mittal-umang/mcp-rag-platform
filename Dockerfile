# --- build stage ---------------------------------------------------------
FROM python:3.11-slim AS build
WORKDIR /app
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

# --- runtime stage -------------------------------------------------------
FROM python:3.11-slim AS runtime
WORKDIR /app
# non-root
RUN useradd --create-home --uid 10001 appuser
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY src ./src
COPY data ./data
USER appuser
EXPOSE 8000
# default entrypoint is the agent; compose overrides command for the mcp-server
CMD ["uvicorn", "src.agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
