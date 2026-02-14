document.addEventListener("DOMContentLoaded", () => {
    fetchReports();
});

async function fetchReports() {
    try {
        const response = await fetch("/api/reports");
        const reports = await response.json();
        renderSummaryStats(reports);
        renderEntries(reports);
    } catch (err) {
        document.getElementById("entries-list").innerHTML =
            '<div class="no-data">Failed to load reports. Is the server running?</div>';
    }
}

function renderSummaryStats(reports) {
    const el = document.getElementById("summary-stats");
    let totalBikes = 0, compliantBikes = 0, totalViolations = 0;

    reports.forEach(r => {
        totalBikes += r.summary?.total_bikes_detected || 0;
        compliantBikes += r.summary?.bikes_with_both_lights || 0;
        totalViolations += (r.violations?.length || 0);
    });

    const rate = totalBikes > 0 ? Math.round((compliantBikes / totalBikes) * 100) : 0;

    el.innerHTML = `
        <div class="stat-card">
            <span class="stat-value">${reports.length}</span>
            <span class="stat-label">Videos Analyzed</span>
        </div>
        <div class="stat-card">
            <span class="stat-value">${totalBikes}</span>
            <span class="stat-label">Bikes Detected</span>
        </div>
        <div class="stat-card">
            <span class="stat-value">${rate}%</span>
            <span class="stat-label">Light Compliance</span>
        </div>
        <div class="stat-card">
            <span class="stat-value">${totalViolations}</span>
            <span class="stat-label">Violations</span>
        </div>
    `;
}

function renderEntries(reports) {
    const el = document.getElementById("entries-list");
    el.innerHTML = "";

    if (reports.length === 0) {
        el.innerHTML = '<div class="no-data">No reports found. Process a video first.</div>';
        return;
    }

    reports.forEach(report => {
        el.appendChild(createEntryCard(report));
    });
}

function createEntryCard(report) {
    const card = document.createElement("div");
    card.className = "entry-card";

    const videoName = report._video_stem || report.video_info?.filename || "Unknown";
    const bikeCount = report.summary?.total_bikes_detected || 0;
    const complianceRate = report.summary?.compliance_rate;
    const processedAt = report.report_metadata?.generated_at
        ? new Date(report.report_metadata.generated_at).toLocaleDateString("en-US", {
            month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit"
        })
        : "Unknown date";

    const timeOfDay = report.video_metadata?.time_of_day || "Unknown";
    const locationName = report.video_metadata?.location?.name || null;

    const hasViolations = (report.violations?.length || 0) > 0;
    const allCompliant = complianceRate === 100;
    let badgeClass, badgeText;
    if (allCompliant) {
        badgeClass = "badge-compliant";
        badgeText = "COMPLIANT";
    } else if (hasViolations) {
        badgeClass = "badge-violation";
        badgeText = "VIOLATION";
    } else {
        badgeClass = "badge-partial";
        badgeText = "PARTIAL";
    }

    const metaParts = [
        `${bikeCount} bike${bikeCount !== 1 ? "s" : ""}`,
        processedAt,
        timeOfDay !== "Unknown" ? capitalizeFirst(timeOfDay) : null,
        locationName
    ].filter(Boolean).join(" &middot; ");

    card.innerHTML = `
        <div class="entry-header" onclick="toggleExpand(this)">
            <div class="entry-info">
                <h3 class="entry-title">${formatVideoName(videoName)}</h3>
                <span class="entry-meta">${metaParts}</span>
            </div>
            <div class="entry-badges">
                <span class="badge ${badgeClass}">${badgeText}</span>
                <span class="chevron">&#9660;</span>
            </div>
        </div>
        <div class="entry-detail">
            <div class="entry-detail-inner">
                ${renderViolations(report.violations || [])}
                <div>
                    <div class="section-label">Cyclist Details</div>
                    ${renderBikeCards(report.bikes || [], report.video_metadata)}
                </div>
                <div>
                    <div class="section-label">Video Analysis</div>
                    ${renderVideoPair(report._files)}
                </div>
                <div>
                    <div class="section-label">Location & Heatmap</div>
                    ${renderMapAndHeatmap(report.video_metadata?.location, report._files)}
                </div>
            </div>
        </div>
    `;

    return card;
}

function toggleExpand(headerEl) {
    const card = headerEl.closest(".entry-card");
    const wasExpanded = card.classList.contains("expanded");
    card.classList.toggle("expanded");

    if (!wasExpanded) {
        card.querySelectorAll("video").forEach(v => {
            v.play().catch(() => {});
        });
    } else {
        card.querySelectorAll("video").forEach(v => {
            v.pause();
            v.currentTime = 0;
        });
    }
}

function renderViolations(violations) {
    if (!violations.length) return "";

    return `
        <div>
            <div class="section-label">Violations</div>
            <div class="violations-list">
                ${violations.map(v => {
                    const severityClass = v.severity === "HIGH" ? "" : "severity-medium";
                    const icon = v.severity === "HIGH" ? "!!" : "!";
                    return `
                        <div class="violation-item ${severityClass}">
                            <span class="violation-icon">${icon}</span>
                            <span>${v.description} (Bike #${v.track_id}) &mdash; ${v.severity} severity</span>
                        </div>`;
                }).join("")}
            </div>
        </div>`;
}

function renderBikeCards(bikes, metadata) {
    if (!bikes.length) return '<p class="no-data">No bikes detected in this video.</p>';

    return `<div class="bike-cards">${bikes.map(bike => {
        const lights = bike.lights || {};
        const hasLights = lights.has_front_light || lights.has_rear_light;
        const compliance = lights.compliance_status || "UNKNOWN";
        const color = bike.color?.primary_color || "unknown";
        const speed = bike.speed?.avg_speed_kmh;
        const depth = bike.depth?.avg_relative_depth;
        const timeOfDay = metadata?.time_of_day || "Unknown";

        const compBadge = compliance === "COMPLIANT"
            ? "badge-compliant" : "badge-violation";

        return `
        <div class="bike-card">
            <div class="bike-card-header">
                <span class="bike-id">Bike #${bike.track_id}</span>
                <span class="badge ${compBadge}">${compliance}</span>
            </div>
            <div class="bike-card-body">
                <div class="detail-row">
                    <span class="detail-label">Lights</span>
                    <span class="detail-value">
                        <span class="light-indicator">
                            <span class="light-dot ${hasLights ? 'on' : 'off'}"></span>
                            ${hasLights ? 'Yes' : 'No'}
                        </span>
                    </span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Bike Color</span>
                    <span class="detail-value">${capitalizeFirst(color)}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Time of Day</span>
                    <span class="detail-value">${capitalizeFirst(timeOfDay)}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Entry ID</span>
                    <span class="detail-value">#${bike.track_id}</span>
                </div>
                ${speed !== undefined ? `
                <div class="detail-row">
                    <span class="detail-label">Avg Speed</span>
                    <span class="detail-value">${speed.toFixed(1)} km/h</span>
                </div>` : ''}
                ${depth !== undefined ? `
                <div class="detail-row">
                    <span class="detail-label">Relative Depth</span>
                    <span class="detail-value">${(depth * 100).toFixed(0)}%</span>
                </div>` : ''}
                <div class="detail-row">
                    <span class="detail-label">Visible Duration</span>
                    <span class="detail-value">${bike.duration_seconds ? bike.duration_seconds.toFixed(2) + 's' : 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Appearances</span>
                    <span class="detail-value">${bike.total_appearances || 'N/A'} frames</span>
                </div>
            </div>
        </div>`;
    }).join("")}</div>`;
}

function renderVideoPair(files) {
    if (!files) return '<p class="no-data">No video files available.</p>';
    const annotated = files.annotated_video;
    const depth = files.depth_video;

    if (!annotated && !depth) return '<p class="no-data">No video files available.</p>';

    return `
    <div class="video-pair">
        ${annotated ? `
        <div class="video-container">
            <label>YOLO Detection</label>
            <video muted loop playsinline preload="metadata">
                <source src="${annotated}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>` : ''}
        ${depth ? `
        <div class="video-container">
            <label>Depth Estimation</label>
            <video muted loop playsinline preload="metadata">
                <source src="${depth}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>` : `
        <div class="video-container">
            <label>Depth Estimation</label>
            <div class="no-data-box">Depth video not available</div>
        </div>`}
    </div>`;
}

function renderMapAndHeatmap(location, files) {
    let mapHtml, heatmapHtml;

    if (location && location.lat && location.lng) {
        const bbox = `${location.lng - 0.005},${location.lat - 0.003},${location.lng + 0.005},${location.lat + 0.003}`;
        const mapUrl = `https://www.openstreetmap.org/export/embed.html?bbox=${bbox}&layer=mapnik&marker=${location.lat},${location.lng}`;
        const displayName = location.name || `${location.lat.toFixed(4)}, ${location.lng.toFixed(4)}`;
        mapHtml = `
        <div class="map-container">
            <label>Location: ${displayName}</label>
            <iframe src="${mapUrl}" frameborder="0" scrolling="no" loading="lazy"></iframe>
        </div>`;
    } else {
        mapHtml = `
        <div class="map-container">
            <label>Recording Location</label>
            <div class="no-data-box">Location not recorded</div>
        </div>`;
    }

    if (files?.heatmap_overlay) {
        heatmapHtml = `
        <div class="heatmap-container">
            <label>Trajectory Heatmap</label>
            <img src="${files.heatmap_overlay}" alt="Trajectory Heatmap" class="heatmap-img" loading="lazy">
        </div>`;
    } else if (files?.heatmap) {
        heatmapHtml = `
        <div class="heatmap-container">
            <label>Trajectory Heatmap</label>
            <img src="${files.heatmap}" alt="Trajectory Heatmap" class="heatmap-img" loading="lazy">
        </div>`;
    } else {
        heatmapHtml = `
        <div class="heatmap-container">
            <label>Trajectory Heatmap</label>
            <div class="no-data-box">Heatmap not available</div>
        </div>`;
    }

    return `<div class="map-heatmap-row">${mapHtml}${heatmapHtml}</div>`;
}

/* === Utilities === */
function capitalizeFirst(str) {
    if (!str) return str;
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function formatVideoName(stem) {
    return stem.replace(/[_-]/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}
