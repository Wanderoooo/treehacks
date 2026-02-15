document.addEventListener("DOMContentLoaded", () => {
    fetchReports();
    initIncidentHeatmap();
    initLiveFeedTimestamp();
    initChat();
    // Auto-refresh reports every 10 seconds for live Jetson uploads
    setInterval(fetchReports, 10000);
});

function initLiveFeedTimestamp() {
    const el = document.getElementById("feed-timestamp");
    if (!el) return;
    function update() {
        const now = new Date();
        const ts = now.toLocaleString("en-US", {
            year: "numeric", month: "2-digit", day: "2-digit",
            hour: "2-digit", minute: "2-digit", second: "2-digit",
            hour12: false
        });
        el.textContent = "REC " + ts;
    }
    update();
    setInterval(update, 1000);
}

function initIncidentHeatmap() {
    const el = document.getElementById("incident-heatmap-global");
    if (!el) return;

    const map = L.map("incident-heatmap-global", { zoomControl: true }).setView([37.4275, -122.1697], 15);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "&copy; OSM",
        maxZoom: 18
    }).addTo(map);

    // Fake incident hotspot data around Stanford campus
    const incidentPoints = [
        // Palm Drive cluster (heaviest — main hotspot)
        [37.4275, -122.1697, 1.0],
        [37.4274, -122.1699, 0.95],
        [37.4276, -122.1695, 0.9],
        [37.4273, -122.1701, 0.85],
        [37.4277, -122.1693, 0.8],
        [37.4272, -122.1698, 0.9],
        [37.4278, -122.1696, 0.85],
        [37.4271, -122.1700, 0.75],
        [37.4279, -122.1694, 0.7],
        [37.4275, -122.1700, 0.8],
        [37.4274, -122.1694, 0.75],
        [37.4276, -122.1701, 0.7],
        // The Oval
        [37.4260, -122.1720, 0.7],
        [37.4258, -122.1718, 0.65],
        [37.4262, -122.1722, 0.6],
        [37.4256, -122.1716, 0.55],
        [37.4264, -122.1724, 0.5],
        [37.4259, -122.1721, 0.6],
        [37.4261, -122.1717, 0.55],
        [37.4257, -122.1723, 0.5],
        // Campus Drive East
        [37.4285, -122.1675, 0.6],
        [37.4283, -122.1678, 0.55],
        [37.4287, -122.1672, 0.5],
        [37.4281, -122.1681, 0.45],
        [37.4289, -122.1669, 0.4],
        [37.4284, -122.1676, 0.5],
        [37.4286, -122.1673, 0.45],
        // Serra Mall / Quad area
        [37.4250, -122.1740, 0.55],
        [37.4248, -122.1738, 0.5],
        [37.4252, -122.1742, 0.45],
        [37.4246, -122.1736, 0.4],
        [37.4254, -122.1744, 0.35],
        [37.4249, -122.1741, 0.45],
        [37.4251, -122.1737, 0.4],
        // Escondido / Stern area
        [37.4240, -122.1710, 0.45],
        [37.4238, -122.1712, 0.4],
        [37.4242, -122.1708, 0.35],
        [37.4236, -122.1714, 0.3],
        [37.4239, -122.1709, 0.35],
        // Galvez St / Stadium area
        [37.4345, -122.1612, 0.4],
        [37.4343, -122.1615, 0.35],
        [37.4347, -122.1609, 0.3],
        [37.4341, -122.1618, 0.25],
        // Lomita Mall
        [37.4270, -122.1680, 0.5],
        [37.4268, -122.1682, 0.45],
        [37.4272, -122.1678, 0.4],
        [37.4266, -122.1684, 0.35],
        [37.4269, -122.1679, 0.4],
        // Lagunita / El Camino edge
        [37.4295, -122.1655, 0.35],
        [37.4293, -122.1658, 0.3],
        [37.4297, -122.1652, 0.25],
        // Junipero Serra Blvd
        [37.4230, -122.1750, 0.4],
        [37.4228, -122.1752, 0.35],
        [37.4232, -122.1748, 0.3],
        [37.4226, -122.1754, 0.25],
        [37.4229, -122.1751, 0.3],
    ];

    L.heatLayer(incidentPoints, {
        radius: 35,
        blur: 20,
        maxZoom: 17,
        max: 1.0,
        minOpacity: 0.4,
        gradient: {0.2: "#ffffb2", 0.4: "#fecc5c", 0.6: "#fd8d3c", 0.8: "#f03b20", 1.0: "#bd0026"}
    }).addTo(map);

    L.marker([37.4275, -122.1697]).addTo(map)
        .bindPopup("Palm Drive, Stanford, CA");
}

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

    if (reports.length === 0) {
        el.innerHTML = '<div class="no-data">No reports found. Process a video first.</div>';
        return;
    }

    // Track which entries are currently expanded so we can preserve state
    const expandedStems = new Set();
    el.querySelectorAll(".entry-card.expanded").forEach(card => {
        const stem = card.dataset.stem;
        if (stem) expandedStems.add(stem);
    });

    // Build a map of existing cards by stem
    const existingCards = {};
    el.querySelectorAll(".entry-card").forEach(card => {
        if (card.dataset.stem) existingCards[card.dataset.stem] = card;
    });

    // Build new card list, reusing expanded cards to avoid closing them
    const newStems = new Set(reports.map(r => r._video_stem || ""));
    const fragment = document.createDocumentFragment();

    reports.forEach(report => {
        const stem = report._video_stem || "";
        if (expandedStems.has(stem) && existingCards[stem]) {
            // Keep expanded card as-is to avoid closing it
            fragment.appendChild(existingCards[stem]);
        } else {
            const card = createEntryCard(report);
            card.dataset.stem = stem;
            fragment.appendChild(card);
        }
    });

    // Remove old cards that no longer exist
    el.innerHTML = "";
    el.appendChild(fragment);
}

function createEntryCard(report) {
    const card = document.createElement("div");
    card.className = "entry-card";
    card.dataset.stem = report._video_stem || "";

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
    let mapHtml;

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

    return `<div class="map-heatmap-row">${mapHtml}<div></div></div>`;
}

/* === Utilities === */
function capitalizeFirst(str) {
    if (!str) return str;
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function formatVideoName(stem) {
    return stem.replace(/[_-]/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

/* === Chat === */
function initChat() {
    const form = document.getElementById("chat-form");
    const input = document.getElementById("chat-input");
    const messages = document.getElementById("chat-messages");
    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = input.value.trim();
        if (!text) return;

        // Add user message
        appendChatMsg("user", text);
        input.value = "";
        input.disabled = true;
        document.getElementById("chat-send").disabled = true;

        // Show typing indicator
        const typingEl = appendChatMsg("bot", "", true);

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({message: text})
            });
            const data = await res.json();
            typingEl.remove();
            appendChatMsg("bot", data.reply || "(no response)");
        } catch (err) {
            typingEl.remove();
            appendChatMsg("bot", "Connection error. Is the server running?");
        }

        input.disabled = false;
        document.getElementById("chat-send").disabled = false;
        input.focus();
    });
}

function appendChatMsg(role, text, typing = false) {
    const messages = document.getElementById("chat-messages");
    const wrapper = document.createElement("div");
    wrapper.className = `chat-msg ${role}${typing ? " chat-typing" : ""}`;
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble";
    bubble.textContent = text;
    wrapper.appendChild(bubble);
    messages.appendChild(wrapper);
    messages.scrollTop = messages.scrollHeight;
    return wrapper;
}
