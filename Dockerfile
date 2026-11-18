# Separate OpenSim pose staging. Not the Orca production web-app.
FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib OPENSIM_JOBS_DIR=/tmp/opensim-jobs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/matplotlib /tmp/opensim-jobs
COPY render_pose/requirements.txt /app/render_pose/requirements.txt
RUN pip install --no-cache-dir -r /app/render_pose/requirements.txt
COPY kinematics.py /app/kinematics.py
COPY analyze_pose.py /app/analyze_pose.py
COPY render_pose/__init__.py /app/render_pose/__init__.py
COPY render_pose/server_app.py /app/render_pose/server_app.py
EXPOSE 10000
CMD ["sh","-c","gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 600 render_pose.server_app:app"]
