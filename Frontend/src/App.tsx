import { useState, useEffect, useRef, useCallback } from "react";

// Design tokens 
const G = "#ffffff";
const G_SILVER = "#86868b";
const G2 = "rgba(255, 255, 255, 0.08)";
const G3 = "rgba(255, 255, 255, 0.03)";
const BG = "#000000";
const GLASS = "rgba(255, 255, 255, 0.03)";
const BORDER = "rgba(255, 255, 255, 0.08)";
const API_URL = "http://localhost:8000";

// Types 
interface Weakness { title: string; detail: string; severity: string; joint: string }
interface Zone { delivery: string; reason: string; risk: string }
interface Drill { name: string; description: string; duration: string; targets: string }
interface Analysis {
    summary: string; player_type: string; overall_score: number;
    scores: Record<string, number>; weaknesses: Weakness[];
    vulnerable_zones: Zone[]; drills: Drill[]; strengths: string[];
    pro_comparison: string; model?: string;
}
interface ApiResult {
    success: boolean; elapsed_seconds: number;
    annotated_image: string;
    features: { joint_angles: Record<string, number | null>; body_metrics: Record<string, number> };
    analysis: Analysis;
}

// Global styles 
const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Bebas+Neue&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:#000000;color:#fff;font-family:'Inter',sans-serif;overflow-x:hidden;-webkit-font-smoothing:antialiased}
::-webkit-scrollbar{width:3px}
::-webkit-scrollbar-thumb{background:#222;border-radius:2px}
.bb{font-family:'Bebas Neue',sans-serif}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(1.6)}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
@keyframes marquee{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
@keyframes fadeUp{from{opacity:0;transform:translateY(32px)}to{opacity:1;transform:translateY(0)}}
@keyframes scan{0%{top:-6%}100%{top:106%}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes glow-pulse{0%,100%{box-shadow:0 0 20px rgba(255,255,255,.15)}50%{box-shadow:0 0 44px rgba(255,255,255,.3)}}
.fuv{opacity:0;transform:translateY(32px);transition:opacity .75s ease,transform .75s ease}
.fuv.in{opacity:1;transform:translateY(0)}
.mq{display:flex;animation:marquee 28s linear infinite}
.spin{animation:spin 1s linear infinite}
.text-gradient{background:linear-gradient(180deg,#ffffff 20%,#86868b 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
`;

// Hooks 
function useFadeUp() {
    const ref = useRef<HTMLDivElement>(null);
    useEffect(() => {
        const el = ref.current; if (!el) return;
        const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) el.classList.add("in"); }, { threshold: 0.12 });
        obs.observe(el);
        return () => obs.disconnect();
    }, []);
    return ref;
}

// Shared primitives 
function Pill({ children, color = G }: { children: React.ReactNode; color?: string }) {
    return (
        <span style={{
            display: "inline-flex", alignItems: "center", gap: 6, padding: "4px 13px",
            borderRadius: 100, border: `1px solid ${color}44`, background: `${color}10`,
            fontSize: 11, fontWeight: 500, letterSpacing: 1, color
        }}>
            <span style={{ width: 5, height: 5, borderRadius: "50%", background: color, animation: "pulse 1.8s ease-in-out infinite" }} />
            {children}
        </span>
    );
}

function GlassCard({ children, style = {}, hover = true, onClick }:
    { children: React.ReactNode; style?: React.CSSProperties; hover?: boolean; onClick?: () => void }) {
    const [hov, setHov] = useState(false);
    return (
        <div onClick={onClick}
            onMouseEnter={() => hover && setHov(true)}
            onMouseLeave={() => hover && setHov(false)}
            style={{
                background: hov ? "rgba(255, 255, 255, 0.06)" : GLASS,
                border: `1px solid ${hov ? "rgba(255, 255, 255, 0.22)" : BORDER}`,
                borderRadius: 16, backdropFilter: "blur(12px)",
                transition: "all .3s ease",
                boxShadow: hov ? "0 8px 32px rgba(255, 255, 255, 0.05)" : "none",
                cursor: onClick ? "pointer" : "default", ...style
            }}>
            {children}
        </div>
    );
}

function Btn({ children, variant = "solid", onClick, style = {}, disabled = false }:
    {
        children: React.ReactNode; variant?: "solid" | "outline"; onClick?: () => void;
        style?: React.CSSProperties; disabled?: boolean
    }) {
    const [hov, setHov] = useState(false);
    const solid = variant === "solid";
    return (
        <button onClick={onClick} disabled={disabled}
            onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
            style={{
                background: solid ? (hov ? "#e8e8ed" : G) : (hov ? "rgba(255, 255, 255, 0.08)" : "transparent"),
                color: solid ? "#000" : "#fff",
                border: solid ? "none" : `1px solid ${hov ? "#ffffff" : "rgba(255, 255, 255, 0.18)"}`,
                borderRadius: 10, padding: "13px 28px", fontSize: 14, fontWeight: 600,
                cursor: disabled ? "not-allowed" : "pointer", fontFamily: "Inter",
                letterSpacing: .2, opacity: disabled ? .5 : 1,
                boxShadow: solid ? (hov ? `0 0 32px rgba(255, 255, 255, .25)` : `0 0 16px rgba(255, 255, 255, .1)`) : "none",
                transition: "all .2s ease", whiteSpace: "nowrap", ...style
            }}>
            {children}
        </button>
    );
}

// Skeleton SVG 
function SkeletonPlayer({ scrollY = 0 }: { scrollY?: number }) {
    const swing = Math.min((scrollY / 400) * 30, 28);
    const J: [number, number][] = [
        [120, 52], [120, 82], [92, 105], [148, 105], [72, 148], [158, 132],
        [58, 188], [172, 162], [120, 168], [103, 168], [137, 168],
        [98, 220], [142, 212], [93, 268], [147, 260],
    ];
    const bones = [[0, 1], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [8, 9], [8, 10], [9, 11], [10, 12], [11, 13], [12, 14]];
    const rad = (swing * Math.PI) / 180;
    const batTipX = J[6][0] - Math.sin(rad) * 72;
    const batTipY = J[6][1] + Math.cos(rad) * 72;
    return (
        <svg viewBox="0 0 240 310" style={{ width: "100%", maxWidth: 320, filter: `drop-shadow(0 0 28px rgba(255,255,255,0.2))` }}>
            <defs>
                <radialGradient id="halo" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor={G} stopOpacity=".18" />
                    <stop offset="100%" stopColor={G} stopOpacity="0" />
                </radialGradient>
            </defs>
            <ellipse cx="120" cy="288" rx="52" ry="7" fill={G} opacity=".07" />
            <circle cx="120" cy="155" r="105" fill="url(#halo)" />
            {bones.map(([a, b], i) => (
                <g key={i}>
                    <line x1={J[a][0]} y1={J[a][1]} x2={J[b][0]} y2={J[b][1]} stroke={G} strokeWidth="5" strokeOpacity=".12" strokeLinecap="round" />
                    <line x1={J[a][0]} y1={J[a][1]} x2={J[b][0]} y2={J[b][1]} stroke={G} strokeWidth="1.5" strokeOpacity=".7" strokeLinecap="round" strokeDasharray="3 2" />
                </g>
            ))}
            <line x1={J[6][0]} y1={J[6][1]} x2={batTipX} y2={batTipY} stroke="#fff" strokeWidth="3.5" strokeLinecap="round" />
            <line x1={J[6][0]} y1={J[6][1]} x2={batTipX} y2={batTipY} stroke={G} strokeWidth="8" strokeLinecap="round" opacity=".15" />
            {J.map(([x, y], i) => (
                <g key={i}>
                    <circle cx={x} cy={y} r={i === 0 ? 14 : 6} fill={G} opacity=".08" />
                    <circle cx={x} cy={y} r={i === 0 ? 9 : 4} fill="#000000" stroke={G} strokeWidth={i === 0 ? 1.8 : 1.2} />
                    <circle cx={x} cy={y} r={i === 0 ? 4.5 : 2.2} fill={G}
                        style={{ animation: `pulse ${1.1 + i * .13}s ease-in-out infinite`, animationDelay: `${i * .07}s` }} />
                </g>
            ))}
            <line x1={J[5][0]} y1={J[5][1]} x2={J[5][0] + 26} y2={J[5][1] - 14} stroke={G} strokeWidth=".6" opacity=".5" />
            <text x={J[5][0] + 28} y={J[5][1] - 16} fill={G} fontSize="7" fontFamily="Inter" opacity=".8">138°</text>
            <line x1={J[11][0]} y1={J[11][1]} x2={J[11][0] - 26} y2={J[11][1] + 6} stroke={G} strokeWidth=".6" opacity=".5" />
            <text x={J[11][0] - 68} y={J[11][1] + 8} fill={G} fontSize="7" fontFamily="Inter" opacity=".8">121°</text>
        </svg>
    );
}

// LANDING PAGE

function Navbar({ scrollY, onCTA }: { scrollY: number; onCTA: () => void }) {
    const solid = scrollY > 60;
    return (
        <nav style={{
            position: "fixed", top: 0, left: 0, right: 0, zIndex: 200,
            background: solid ? "rgba(0,0,0,0.85)" : "transparent",
            backdropFilter: solid ? "blur(24px)" : "none",
            borderBottom: solid ? `1px solid ${BORDER}` : "none",
            transition: "all .35s ease", padding: "0 48px", height: 62,
            display: "flex", alignItems: "center", justifyContent: "space-between"
        }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <svg width="26" height="26" viewBox="0 0 28 28">
                    <circle cx="14" cy="14" r="13" fill="none" stroke={G} strokeWidth="1.5" />
                    <path d="M8 14 Q14 6 20 14 Q14 22 8 14Z" fill={G} opacity=".85" />
                    <circle cx="14" cy="14" r="2.5" fill="#fff" />
                </svg>
                <span className="bb" style={{ fontSize: 21, letterSpacing: 2 }}>CrickIQ</span>
            </div>
            <div style={{ display: "flex", gap: 32, alignItems: "center" }}>
                {["Features", "Pipeline"].map(l => (
                    <a key={l} href={`#${l.toLowerCase()}`}
                        style={{ color: "rgba(255,255,255,0.5)", fontSize: 13, textDecoration: "none", transition: "color .2s" }}
                        onMouseEnter={e => (e.currentTarget.style.color = "#fff")}
                        onMouseLeave={e => (e.currentTarget.style.color = "rgba(255,255,255,0.5)")}>{l}</a>
                ))}
                <Btn onClick={onCTA} style={{ padding: "9px 20px", fontSize: 13 }}>Get Early Access</Btn>
            </div>
        </nav>
    );
}

function Hero({ scrollY, onCTA }: { scrollY: number; onCTA: () => void }) {
    return (
        <section style={{
            minHeight: "100vh", display: "flex", alignItems: "center",
            padding: "120px 80px 80px", position: "relative", overflow: "hidden"
        }}>
            <div style={{
                position: "absolute", inset: 0,
                backgroundImage: `linear-gradient(rgba(255,255,255,0.015) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,0.015) 1px,transparent 1px)`,
                backgroundSize: "56px 56px"
            }} />
            <div style={{
                position: "absolute", left: 0, right: 0, height: 1,
                background: `linear-gradient(90deg,transparent,rgba(255,255,255,0.18),transparent)`,
                animation: "scan 7s linear infinite", pointerEvents: "none"
            }} />
            <div style={{
                position: "absolute", top: "20%", right: "20%", width: 560, height: 560,
                background: `radial-gradient(circle,rgba(255,255,255,0.03) 0%,transparent 70%)`,
                transform: `translateY(${scrollY * .08}px)`, pointerEvents: "none"
            }} />

            <div style={{ flex: 1, maxWidth: 620, position: "relative", zIndex: 2 }}>
                <h1 className="bb" style={{
                    fontSize: "clamp(68px,8.5vw,116px)", lineHeight: .92,
                    letterSpacing: 2, marginBottom: 26,
                    animation: "fadeUp .9s ease forwards", opacity: 0, animationDelay: ".1s"
                }}>
                    <span style={{ display: "block", color: "#ffffff" }}>TRAIN</span>
                    <span className="text-gradient" style={{ display: "block" }}>SMARTER.</span>
                    <span style={{ display: "block", color: "#ffffff" }}>WIN</span>
                    <span className="text-gradient" style={{ display: "block" }}>BIGGER.</span>
                </h1>
                <p style={{
                    fontSize: 16, color: "rgba(255,255,255,0.55)", lineHeight: 1.75,
                    maxWidth: 460, marginBottom: 38,
                    animation: "fadeUp .9s ease forwards", opacity: 0, animationDelay: ".28s"
                }}>
                    Upload your batting or bowling footage. Our on-device AI detects
                    biomechanical weaknesses, tactical vulnerabilities, and generates
                    personalised drill plans — powered by CrickLM, a transformer
                    model built from scratch.
                </p>
                <div style={{
                    display: "flex", gap: 12,
                    animation: "fadeUp .9s ease forwards", opacity: 0, animationDelay: ".44s"
                }}>
                    <Btn onClick={onCTA}>Analyze Performance →</Btn>
                    <Btn variant="outline">▷ Watch Demo</Btn>
                </div>
                <div style={{
                    display: "flex", gap: 36, marginTop: 52, paddingTop: 32,
                    borderTop: `1px solid ${BORDER}`,
                    animation: "fadeUp .9s ease forwards", opacity: 0, animationDelay: ".58s"
                }}>
                    {[["98%", "Accuracy"], ["2.4s", "Analysis"], ["10M", "Parameters"]].map(([n, l]) => (
                        <div key={l}>
                            <div className="bb" style={{ fontSize: 38, color: "#ffffff", letterSpacing: 1 }}>{n}</div>
                            <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", letterSpacing: .5, marginTop: 2 }}>{l}</div>
                        </div>
                    ))}
                </div>
            </div>

            <div style={{
                flex: 1, display: "flex", justifyContent: "center", alignItems: "center",
                zIndex: 2, animation: "float 4s ease-in-out infinite",
                transform: `translateY(${-scrollY * .05}px)`
            }}>
                <SkeletonPlayer scrollY={scrollY} />
            </div>
        </section>
    );
}

// TrustedBy section removed

function Pipeline() {
    const ref = useFadeUp();
    const steps = [
        { icon: "⬆", label: "Input layer", sub: "Video (.mp4/.mov) or Image (.jpg/.png) of batsman or bowler", color: "#ffffff" },
        { icon: "⚙", label: "Preprocessing & pose estimation", sub: "Frame extraction · MediaPipe keypoints · 33-point skeleton", color: "#e8e8ed" },
        { icon: "⬡", label: "Feature extraction engine", sub: "Joint angles · Foot balance · Stance width · Wrist height", color: "#d2d2d7" },
        { icon: "✦", label: "CrickLM analysis", sub: "Local transformer (10M params, built from scratch) → weakness + vulnerability detection", color: "#86868b" },
        { icon: "↓", label: "Output layer", sub: "Weakness report · Annotated skeleton image · Drill plan · Score", color: "#6e6e73" },
    ];
    return (
        <section id="pipeline" style={{ padding: "120px 80px" }}>
            <div ref={ref} className="fuv" style={{ textAlign: "center", marginBottom: 72 }}>
                <p style={{ fontSize: 10, color: G_SILVER, letterSpacing: 2, marginBottom: 10 }}>HOW IT WORKS</p>
                <h2 className="bb" style={{ fontSize: "clamp(44px,5.5vw,68px)", letterSpacing: 1 }}>The Intelligence Pipeline</h2>
            </div>
            <div style={{ maxWidth: 680, margin: "0 auto", position: "relative" }}>
                <div style={{
                    position: "absolute", left: 30, top: 0, bottom: 0, width: 1,
                    background: `linear-gradient(to bottom,transparent,rgba(255,255,255,0.15),transparent)`
                }} />
                {steps.map((s, i) => {
                    const r = useFadeUp();
                    return (
                        <div key={i} ref={r} className="fuv">
                            <div style={{ display: "flex", gap: 20, marginBottom: 4 }}>
                                <div style={{
                                    width: 60, height: 60, flexShrink: 0, borderRadius: 14,
                                    background: GLASS, border: `1px solid ${s.color}33`,
                                    display: "flex", alignItems: "center", justifyContent: "center",
                                    fontSize: 20, color: s.color, backdropFilter: "blur(10px)",
                                    boxShadow: `0 0 18px ${s.color}18`, position: "relative", zIndex: 1
                                }}>{s.icon}</div>
                                <GlassCard style={{ flex: 1, padding: "16px 20px" }}>
                                    <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>{s.label}</div>
                                    <div style={{ fontSize: 12, color: "rgba(255,255,255,0.42)", lineHeight: 1.6 }}>{s.sub}</div>
                                </GlassCard>
                            </div>
                            {i < steps.length - 1 && <div style={{
                                width: 1, height: 28,
                                background: `${s.color}35`, marginLeft: 29
                            }} />}
                        </div>
                    );
                })}
            </div>
        </section>
    );
}

function Features() {
    const ref = useFadeUp();
    const cards = [
        { icon: "01", t: "AI Biomechanics Engine", d: "33-point MediaPipe skeleton with joint angle computation across all cricket movements." },
        { icon: "02", t: "Batting Weakness Detection", d: "Identifies playing across the line, late footwork, head fall, and closed bat face." },
        { icon: "03", t: "Bowling Action Breakdown", d: "Front-on vs side-on, illegal action flags, release height & front knee analysis." },
        { icon: "04", t: "Vulnerability Prediction", d: "AI maps your weaknesses to the specific delivery most likely to dismiss you." },
        { icon: "05", t: "CrickLM — Built from Scratch", d: "10M parameter transformer trained on cricket commentary & coaching text. No external API." },
        { icon: "06", t: "Session History", d: "Track biomechanical progress across sessions with a Firebase-backed coach dashboard." },
    ];
    return (
        <section id="features" style={{ padding: "120px 80px", background: "rgba(255,255,255,.008)" }}>
            <div ref={ref} className="fuv" style={{ textAlign: "center", marginBottom: 60 }}>
                <p style={{ fontSize: 10, color: G_SILVER, letterSpacing: 2, marginBottom: 10 }}>CAPABILITIES</p>
                <h2 className="bb" style={{ fontSize: "clamp(44px,5.5vw,68px)", letterSpacing: 1 }}>Built for elite performance</h2>
            </div>
            <div style={{
                display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(285px,1fr))",
                gap: 14, maxWidth: 1060, margin: "0 auto"
            }}>
                {cards.map((c, i) => {
                    const r = useFadeUp();
                    return (
                        <div key={i} ref={r} className="fuv" style={{ transitionDelay: `${i * .07}s` }}>
                            <GlassCard style={{ padding: "26px", height: "100%" }}>
                                <div style={{
                                    fontFamily: "'Bebas Neue', sans-serif", fontSize: 13,
                                    color: "rgba(255,255,255,0.2)", letterSpacing: 2, marginBottom: 16
                                }}>{c.icon}</div>
                                <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>{c.t}</div>
                                <div style={{ fontSize: 13, color: "rgba(255,255,255,0.45)", lineHeight: 1.7 }}>{c.d}</div>
                            </GlassCard>
                        </div>
                    );
                })}
            </div>
        </section>
    );
}


function FinalCTA({ onCTA }: { onCTA: () => void }) {
    const ref = useFadeUp();
    return (
        <section style={{ padding: "160px 80px", textAlign: "center", position: "relative", overflow: "hidden" }}>
            <div style={{
                position: "absolute", inset: 0,
                backgroundImage: `radial-gradient(circle at center,rgba(255,255,255,0.03) 0%,transparent 60%),
          linear-gradient(rgba(255,255,255,0.015) 1px,transparent 1px),
          linear-gradient(90deg,rgba(255,255,255,0.015) 1px,transparent 1px)`,
                backgroundSize: "auto,44px 44px,44px 44px"
            }} />
            <div ref={ref} className="fuv">
                <h2 className="bb" style={{
                    fontSize: "clamp(52px,7.5vw,96px)",
                    letterSpacing: 1.5, lineHeight: .95, marginBottom: 26
                }}>
                    Your breakthrough<br /><span className="text-gradient">starts here.</span>
                </h2>
                <p style={{
                    fontSize: 16, color: "rgba(255,255,255,.45)",
                    maxWidth: 400, margin: "0 auto 40px"
                }}>
                    Upload your first image free. No signup. Analysis in under 3 seconds.
                </p>
                <Btn onClick={onCTA}
                    style={{ padding: "15px 44px", fontSize: 15, animation: "glow-pulse 2.5s ease-in-out infinite" }}>
                    Start Free Analysis →
                </Btn>
            </div>
        </section>
    );
}

function Footer() {
    return (
        <footer style={{
            borderTop: `1px solid ${BORDER}`, padding: "44px 80px",
            display: "flex", justifyContent: "space-between", alignItems: "center",
            flexWrap: "wrap", gap: 20
        }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <svg width="20" height="20" viewBox="0 0 28 28">
                    <circle cx="14" cy="14" r="13" fill="none" stroke={G} strokeWidth="1.5" />
                    <path d="M8 14 Q14 6 20 14 Q14 22 8 14Z" fill={G} opacity=".85" />
                    <circle cx="14" cy="14" r="2.5" fill="#fff" />
                </svg>
                <span className="bb" style={{ fontSize: 17, letterSpacing: 2 }}>CrickIQ</span>
                <span style={{ fontSize: 10, color: "rgba(255,255,255,.2)", marginLeft: 6 }}>© 2025</span>
            </div>
            <div style={{ display: "flex", gap: 24 }}>
                {["Product", "Docs", "API", "Privacy", "Contact"].map(l => (
                    <a key={l} href="#"
                        style={{ fontSize: 12, color: "rgba(255,255,255,.3)", textDecoration: "none", transition: "color .2s" }}
                        onMouseEnter={e => (e.currentTarget.style.color = "#fff")}
                        onMouseLeave={e => (e.currentTarget.style.color = "rgba(255,255,255,.3)")}>{l}</a>
                ))}
            </div>
            <div style={{ fontSize: 11, color: "rgba(255,255,255,.18)" }}>
                Built by A Rohan · CrickLM (10M params, from scratch)
            </div>
        </footer>
    );
}

// ANALYSIS APP

function ScoreRing({ value, size = 64, label }: { value: number; size?: number; label: string }) {
    const r = size / 2 - 6;
    const circ = 2 * Math.PI * r;
    const offset = circ - (value / 100) * circ;
    const color = value >= 80 ? G : value >= 60 ? "#fb923c" : "#ef4444";
    return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 5 }}>
            <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
                <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,.07)" strokeWidth={4} />
                <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={4}
                    strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
                    style={{ transition: "stroke-dashoffset 1.2s ease", filter: `drop-shadow(0 0 4px ${color}88)` }} />
                <text x="50%" y="50%" textAnchor="middle" dominantBaseline="central"
                    fill={color} fontSize={13} fontWeight="600" fontFamily="Inter"
                    style={{ transform: `rotate(90deg)`, transformOrigin: `${size / 2}px ${size / 2}px` }}>{value}</text>
            </svg>
            <span style={{ fontSize: 9, color: "rgba(255,255,255,.4)", letterSpacing: .5, textTransform: "uppercase" }}>{label}</span>
        </div>
    );
}

function UploadZone({ onFile, loading }: { onFile: (f: File) => void; loading: boolean }) {
    const [drag, setDrag] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);
    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault(); setDrag(false);
        const f = e.dataTransfer.files[0]; if (f) onFile(f);
    }, [onFile]);
    return (
        <div onDragOver={e => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={handleDrop}
            onClick={() => !loading && inputRef.current?.click()}
            style={{
                border: `2px dashed ${drag ? G : "rgba(255,255,255,0.12)"}`,
                borderRadius: 20, padding: "52px 32px", textAlign: "center",
                cursor: loading ? "wait" : "pointer",
                background: drag ? G2 : G3, transition: "all .25s ease",
                boxShadow: drag ? `0 0 32px rgba(255,255,255,.08)` : "none"
            }}>
            <input ref={inputRef} type="file" accept="image/*" style={{ display: "none" }}
                onChange={e => { const f = e.target.files?.[0]; if (f) onFile(f); }} />
            {loading ? (
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
                    <svg className="spin" width="36" height="36" viewBox="0 0 36 36">
                        <circle cx="18" cy="18" r="14" fill="none" stroke="rgba(255,255,255,.1)" strokeWidth="3" />
                        <circle cx="18" cy="18" r="14" fill="none" stroke={G} strokeWidth="3"
                            strokeDasharray="88" strokeDashoffset="66" strokeLinecap="round" />
                    </svg>
                    <p style={{ color: G, fontSize: 14, fontWeight: 500 }}>CrickLM analyzing…</p>
                    <p style={{ color: "rgba(255,255,255,.38)", fontSize: 12 }}>MediaPipe pose extraction + local transformer</p>
                </div>
            ) : (
                <>
                    <svg width="40" height="40" viewBox="0 0 40 40" style={{ marginBottom: 16, opacity: 0.5 }}>
                        <rect x="4" y="26" width="32" height="2" rx="1" fill="#fff" />
                        <path d="M20 6 L20 22 M13 13 L20 6 L27 13" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                    </svg>
                    <p style={{ fontSize: 15, fontWeight: 500, marginBottom: 8 }}>Drop image here or click to upload</p>
                    <p style={{ fontSize: 12, color: "rgba(255,255,255,.38)", marginBottom: 20 }}>JPG · PNG · WEBP · max 15 MB</p>
                    <Pill>Batting & bowling stance analysis</Pill>
                </>
            )}
        </div>
    );
}

const SEV_COLOR: Record<string, string> = {
    High: "#ef4444", Critical: "#ef4444", Med: "#fb923c", Low: G
};

function ResultPanel({ result }: { result: ApiResult }) {
    const [tab, setTab] = useState(0);
    const { analysis, annotated_image, features } = result;
    const tabs = ["Weaknesses", "Vulnerable Zones", "Drill Plan", "Metrics"];

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {/* Score header */}
            <GlassCard hover={false} style={{ padding: "24px" }}>
                <div style={{
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    flexWrap: "wrap", gap: 16
                }}>
                    <div>
                        <div style={{ fontSize: 10, color: G, letterSpacing: 1.5, marginBottom: 4 }}>OVERALL SCORE</div>
                        <div className="bb" style={{ fontSize: 58, color: G, letterSpacing: 1, lineHeight: 1 }}>
                            {analysis.overall_score}
                        </div>
                        <div style={{ fontSize: 11, color: "rgba(255,255,255,.35)", marginTop: 4 }}>
                            {analysis.player_type?.toUpperCase()} · {analysis.model || "CrickLM"}
                        </div>
                    </div>
                    <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
                        {Object.entries(analysis.scores || {}).map(([k, v]) => (
                            <ScoreRing key={k} value={v} label={k.replace("_", " ")} />
                        ))}
                    </div>
                </div>
                {analysis.summary && (
                    <p style={{
                        fontSize: 13, color: "rgba(255,255,255,.5)", lineHeight: 1.7,
                        marginTop: 16, paddingTop: 16, borderTop: `1px solid ${BORDER}`
                    }}>
                        {analysis.summary}
                    </p>
                )}
            </GlassCard>

            {/* Annotated image */}
            {annotated_image && (
                <GlassCard hover={false} style={{ overflow: "hidden" }}>
                    <div style={{
                        fontSize: 10, color: "rgba(255,255,255,.3)",
                        letterSpacing: 1.5, padding: "14px 20px 0"
                    }}>MEDIAPIPE SKELETON OVERLAY</div>
                    <img src={annotated_image} alt="Annotated"
                        style={{
                            width: "100%", display: "block", borderRadius: "0 0 14px 14px",
                            maxHeight: 340, objectFit: "contain", background: "#0c0c0c"
                        }} />
                </GlassCard>
            )}

            {/* Tabs */}
            <GlassCard hover={false}>
                <div style={{ display: "flex", borderBottom: `1px solid ${BORDER}`, padding: "0 8px" }}>
                    {tabs.map((t, i) => (
                        <button key={i} onClick={() => setTab(i)}
                            style={{
                                background: "none", border: "none", padding: "13px 14px", fontSize: 12,
                                color: tab === i ? "#fff" : "rgba(255,255,255,.35)",
                                borderBottom: tab === i ? `2px solid ${G}` : "2px solid transparent",
                                cursor: "pointer", fontFamily: "Inter", transition: "color .2s",
                                whiteSpace: "nowrap"
                            }}>{t}</button>
                    ))}
                </div>

                <div style={{ padding: "18px" }}>
                    {/* Weaknesses */}
                    {tab === 0 && (
                        <div>
                            {(analysis.weaknesses || []).map((w, i) => (
                                <div key={i} style={{
                                    display: "flex", gap: 12, padding: "12px 14px",
                                    background: GLASS, border: `1px solid ${BORDER}`, borderRadius: 10, marginBottom: 8,
                                    transition: "border-color .2s"
                                }}
                                    onMouseEnter={e => (e.currentTarget.style.borderColor = `${G}30`)}
                                    onMouseLeave={e => (e.currentTarget.style.borderColor = BORDER)}>
                                    <div style={{
                                        width: 34, height: 34, borderRadius: 8,
                                        background: "rgba(239,68,68,.08)", display: "flex",
                                        alignItems: "center", justifyContent: "center", flexShrink: 0, fontSize: 14
                                    }}>⚠</div>
                                    <div style={{ flex: 1 }}>
                                        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 2 }}>{w.title}</div>
                                        <div style={{ fontSize: 11, color: "rgba(255,255,255,.42)", lineHeight: 1.6 }}>{w.detail}</div>
                                        {w.joint && <div style={{ fontSize: 10, color: "rgba(255,255,255,.22)", marginTop: 3 }}>Joint: {w.joint}</div>}
                                    </div>
                                    <div style={{
                                        fontSize: 9, fontWeight: 700, letterSpacing: .5,
                                        color: SEV_COLOR[w.severity] || G,
                                        background: `${SEV_COLOR[w.severity] || G}15`,
                                        padding: "2px 8px", borderRadius: 6, alignSelf: "flex-start", flexShrink: 0
                                    }}>
                                        {w.severity?.toUpperCase()}
                                    </div>
                                </div>
                            ))}
                            {(analysis.strengths || []).length > 0 && (
                                <div style={{
                                    marginTop: 14, padding: "14px", background: G2,
                                    borderRadius: 10, border: `1px solid ${G}33`
                                }}>
                                    <div style={{
                                        fontSize: 10, color: G, fontWeight: 600,
                                        marginBottom: 8, letterSpacing: .5
                                    }}>STRENGTHS DETECTED</div>
                                    {analysis.strengths.map((s, i) => (
                                        <div key={i} style={{
                                            display: "flex", gap: 8, marginBottom: 5,
                                            fontSize: 12, color: "rgba(255,255,255,.6)"
                                        }}>
                                            <span style={{ color: G, flexShrink: 0 }}>✓</span>{s}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Vulnerable zones */}
                    {tab === 1 && (
                        <div>
                            {(analysis.vulnerable_zones || []).map((z, i) => (
                                <div key={i} style={{
                                    display: "flex", gap: 12, padding: "12px 14px",
                                    background: GLASS, border: `1px solid ${BORDER}`, borderRadius: 10, marginBottom: 8,
                                    transition: "border-color .2s"
                                }}
                                    onMouseEnter={e => (e.currentTarget.style.borderColor = `${G}30`)}
                                    onMouseLeave={e => (e.currentTarget.style.borderColor = BORDER)}>
                                    <div style={{
                                        width: 34, height: 34, borderRadius: 8,
                                        background: "rgba(239,68,68,.08)", display: "flex",
                                        alignItems: "center", justifyContent: "center", flexShrink: 0,
                                        fontSize: 14, color: "rgba(239,68,68,0.7)", fontWeight: 700
                                    }}>◈</div>
                                    <div style={{ flex: 1 }}>
                                        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 2 }}>{z.delivery}</div>
                                        <div style={{ fontSize: 11, color: "rgba(255,255,255,.42)", lineHeight: 1.6 }}>{z.reason}</div>
                                    </div>
                                    <div style={{
                                        fontSize: 9, fontWeight: 700, letterSpacing: .5,
                                        color: SEV_COLOR[z.risk] || "#fb923c",
                                        background: `${SEV_COLOR[z.risk] || "#fb923c"}15`,
                                        padding: "2px 8px", borderRadius: 6, alignSelf: "flex-start", flexShrink: 0
                                    }}>
                                        {z.risk?.toUpperCase()}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Drills */}
                    {tab === 2 && (
                        <div>
                            {(analysis.drills || []).map((d, i) => (
                                <div key={i} style={{
                                    padding: "14px", background: GLASS,
                                    border: `1px solid ${BORDER}`, borderRadius: 10, marginBottom: 8,
                                    transition: "border-color .2s"
                                }}
                                    onMouseEnter={e => (e.currentTarget.style.borderColor = `${G}30`)}
                                    onMouseLeave={e => (e.currentTarget.style.borderColor = BORDER)}>
                                    <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 6 }}>
                                        <div style={{
                                            width: 26, height: 26, borderRadius: 7, background: G2,
                                            border: `1px solid ${G}44`, display: "flex", alignItems: "center",
                                            justifyContent: "center", fontSize: 11, color: G, fontWeight: 700, flexShrink: 0
                                        }}>{i + 1}</div>
                                        <div style={{ fontSize: 13, fontWeight: 600 }}>{d.name}</div>
                                    </div>
                                    <div style={{
                                        fontSize: 12, color: "rgba(255,255,255,.45)",
                                        lineHeight: 1.65, marginLeft: 36
                                    }}>{d.description}</div>
                                    {(d.duration || d.targets) && (
                                        <div style={{ display: "flex", gap: 10, marginLeft: 36, marginTop: 8, flexWrap: "wrap" }}>
                                            {d.duration && <span style={{
                                                fontSize: 10, color: "rgba(255,255,255,.3)",
                                                background: "rgba(255,255,255,.04)", padding: "2px 8px", borderRadius: 5
                                            }}>{d.duration}</span>}
                                            {d.targets && <span style={{
                                                fontSize: 10, color: "rgba(255,255,255,.3)",
                                                background: "rgba(255,255,255,.04)", padding: "2px 8px", borderRadius: 5
                                            }}>{d.targets}</span>}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Metrics */}
                    {tab === 3 && (
                        <div>
                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 14 }}>
                                {Object.entries(features?.joint_angles || {})
                                    .filter(([, v]) => v !== null)
                                    .map(([k, v]) => (
                                        <div key={k} style={{
                                            padding: "10px 12px", background: GLASS,
                                            border: `1px solid ${BORDER}`, borderRadius: 8
                                        }}>
                                            <div style={{ fontSize: 10, color: "rgba(255,255,255,.35)", marginBottom: 2 }}>
                                                {k.replace(/_/g, " ").toUpperCase()}</div>
                                            <div style={{ fontSize: 16, fontWeight: 600, color: G }}>{v}°</div>
                                        </div>
                                    ))}
                            </div>
                            {Object.keys(features?.body_metrics || {}).length > 0 && (
                                <div>
                                    <div style={{
                                        fontSize: 10, color: "rgba(255,255,255,.3)",
                                        letterSpacing: 1, marginBottom: 8
                                    }}>BODY METRICS</div>
                                    {Object.entries(features.body_metrics).map(([k, v]) => (
                                        <div key={k} style={{
                                            display: "flex", justifyContent: "space-between",
                                            padding: "8px 0", borderBottom: `1px solid rgba(255,255,255,.04)`
                                        }}>
                                            <span style={{ fontSize: 12, color: "rgba(255,255,255,.4)" }}>
                                                {k.replace(/_/g, " ")}</span>
                                            <span style={{ fontSize: 12, fontWeight: 500 }}>
                                                {typeof v === "number" ? v.toFixed(2) : v}</span>
                                        </div>
                                    ))}
                                </div>
                            )}
                            {analysis.pro_comparison && (
                                <div style={{
                                    marginTop: 14, padding: "12px 14px", background: G2,
                                    border: `1px solid ${G}33`, borderRadius: 10
                                }}>
                                    <div style={{ fontSize: 10, color: G, letterSpacing: 1, marginBottom: 6 }}>PRO COMPARISON</div>
                                    <div style={{ fontSize: 12, color: "rgba(255,255,255,.65)", lineHeight: 1.7 }}>
                                        {analysis.pro_comparison}</div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </GlassCard>
        </div>
    );
}

function AnalysisApp({ onBack }: { onBack: () => void }) {
    const [playerType, setPlayerType] = useState<"batsman" | "bowler">("batsman");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<ApiResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [preview, setPreview] = useState<string | null>(null);

    const handleFile = useCallback(async (file: File) => {
        setError(null); setResult(null);
        setPreview(URL.createObjectURL(file));
        setLoading(true);
        const fd = new FormData();
        fd.append("file", file);
        fd.append("player_type", playerType);
        try {
            const res = await fetch(`${API_URL}/analyze`, { method: "POST", body: fd });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Analysis failed");
            setResult(data);
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : "Unknown error");
        } finally {
            setLoading(false);
        }
    }, [playerType]);

    const reset = () => { setResult(null); setError(null); setPreview(null); };

    return (
        <div style={{ minHeight: "100vh", background: BG }}>
            {/* Sticky navbar */}
            <div style={{
                position: "sticky", top: 0, zIndex: 100,
                background: "rgba(8,8,8,.95)", backdropFilter: "blur(20px)",
                borderBottom: `1px solid ${BORDER}`, padding: "0 32px", height: 58,
                display: "flex", alignItems: "center", justifyContent: "space-between"
            }}>
                <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                    <button onClick={onBack}
                        style={{
                            background: "none", border: `1px solid ${BORDER}`, borderRadius: 8,
                            padding: "5px 12px", fontSize: 12, color: "rgba(255,255,255,.5)",
                            cursor: "pointer", fontFamily: "Inter", transition: "all .2s"
                        }}
                        onMouseEnter={e => { e.currentTarget.style.borderColor = G; e.currentTarget.style.color = G; }}
                        onMouseLeave={e => { e.currentTarget.style.borderColor = BORDER; e.currentTarget.style.color = "rgba(255,255,255,.5)"; }}>
                        ← Back
                    </button>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <svg width="20" height="20" viewBox="0 0 28 28">
                            <circle cx="14" cy="14" r="13" fill="none" stroke={G} strokeWidth="1.5" />
                            <path d="M8 14 Q14 6 20 14 Q14 22 8 14Z" fill={G} opacity=".85" />
                            <circle cx="14" cy="14" r="2.5" fill="#fff" />
                        </svg>
                        <span className="bb" style={{ fontSize: 18, letterSpacing: 2 }}>CrickIQ</span>
                        <span style={{ fontSize: 11, color: "rgba(255,255,255,.25)" }}>Analysis Studio</span>
                    </div>
                </div>
            </div>

            <div style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 32px 60px" }}>
                {/* Player type */}
                {!result && (
                    <div style={{ marginBottom: 24 }}>
                        <div style={{ fontSize: 11, color: "rgba(255,255,255,.35)", letterSpacing: 1, marginBottom: 10 }}>
                            PLAYER TYPE</div>
                        <div style={{ display: "flex", gap: 8 }}>
                            {(["batsman", "bowler"] as const).map(t => (
                                <button key={t} onClick={() => setPlayerType(t)}
                                    style={{
                                        padding: "8px 22px", borderRadius: 9,
                                        border: `1px solid ${playerType === t ? G : BORDER}`,
                                        background: playerType === t ? G2 : GLASS,
                                        color: playerType === t ? G : "rgba(255,255,255,.5)",
                                        fontSize: 13, fontWeight: 500, cursor: "pointer",
                                        fontFamily: "Inter", transition: "all .2s", backdropFilter: "blur(8px)"
                                    }}>
                                    {t === "batsman" ? "Batsman" : "Bowler"}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {/* Error */}
                {error && (
                    <div style={{
                        padding: "14px 18px", background: "rgba(239,68,68,.08)",
                        border: "1px solid rgba(239,68,68,.25)", borderRadius: 12, marginBottom: 20,
                        display: "flex", gap: 12, alignItems: "flex-start"
                    }}>
                        <span style={{ fontSize: 16, flexShrink: 0 }}>⚠</span>
                        <div>
                            <div style={{ fontSize: 13, fontWeight: 600, color: "#ef4444", marginBottom: 3 }}>
                                Analysis failed</div>
                            <div style={{ fontSize: 12, color: "rgba(255,255,255,.5)" }}>{error}</div>
                            {error.includes("fetch") && (
                                <div style={{ fontSize: 11, color: "rgba(255,255,255,.3)", marginTop: 4 }}>
                                    Make sure backend is running:{" "}
                                    <code style={{ color: G }}>uvicorn main:app --port 8000</code>
                                </div>
                            )}
                        </div>
                        <button onClick={reset}
                            style={{
                                marginLeft: "auto", background: "none", border: "none",
                                color: "rgba(255,255,255,.3)", cursor: "pointer", fontSize: 16
                            }}>✕</button>
                    </div>
                )}

                {/* Main content */}
                {!result ? (
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, alignItems: "start" }}>
                        <div>
                            <UploadZone onFile={handleFile} loading={loading} />
                            {preview && !loading && (
                                <div style={{ marginTop: 12, position: "relative" }}>
                                    <img src={preview} alt="preview"
                                        style={{
                                            width: "100%", borderRadius: 12, maxHeight: 260,
                                            objectFit: "contain", background: "#0c0c0c"
                                        }} />
                                    <button onClick={reset}
                                        style={{
                                            position: "absolute", top: 8, right: 8,
                                            background: "rgba(0,0,0,.7)", border: "none", color: "#fff",
                                            borderRadius: 6, padding: "3px 8px", cursor: "pointer", fontSize: 12
                                        }}>✕</button>
                                </div>
                            )}
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                            <GlassCard hover={false} style={{ padding: "22px" }}>
                                <div style={{
                                    fontSize: 12, color: G, fontWeight: 600,
                                    letterSpacing: .5, marginBottom: 16
                                }}>WHAT YOU'LL GET</div>
                                {[
                                    ["01", "Skeleton overlay", "33-point MediaPipe pose mapped onto your image"],
                                    ["02", "Joint angles", "Elbow, knee, shoulder, hip angles measured precisely"],
                                    ["03", "Weakness detection", "CrickLM identifies biomechanical flaws in your stance"],
                                    ["04", "Vulnerable zones", "Deliveries most likely to dismiss you based on your gaps"],
                                    ["05", "Drill plan", "3–4 targeted drills with duration and focus areas"],
                                ].map(([icon, title, desc]) => (
                                    <div key={String(title)} style={{ display: "flex", gap: 12, marginBottom: 14 }}>
                                        <span style={{
                                            fontFamily: "'Bebas Neue', sans-serif", fontSize: 11,
                                            color: "rgba(255,255,255,0.22)", flexShrink: 0,
                                            marginTop: 2, letterSpacing: 1, minWidth: 20
                                        }}>{icon}</span>
                                        <div>
                                            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 2 }}>{title}</div>
                                            <div style={{ fontSize: 11, color: "rgba(255,255,255,.38)", lineHeight: 1.6 }}>{desc}</div>
                                        </div>
                                    </div>
                                ))}
                            </GlassCard>
                            <GlassCard hover={false} style={{ padding: "16px 20px" }}>
                                <div style={{ fontSize: 10, color: "rgba(255,255,255,0.25)", letterSpacing: 1.5, marginBottom: 6 }}>TIP</div>
                                <div style={{ fontSize: 12, color: "rgba(255,255,255,.4)", lineHeight: 1.7 }}>
                                    <strong style={{ color: "rgba(255,255,255,.7)" }}>Best results:</strong>{" "}
                                    Clear side-on or front-on full-body shot. Natural light, plain background.
                                </div>
                            </GlassCard>
                        </div>
                    </div>
                ) : (
                    <div>
                        <div style={{
                            display: "flex", justifyContent: "space-between",
                            alignItems: "center", marginBottom: 20
                        }}>
                            <div>
                                <h2 style={{ fontSize: 20, fontWeight: 600 }}>Analysis complete</h2>
                                <p style={{ fontSize: 12, color: "rgba(255,255,255,.38)", marginTop: 2 }}>
                                    {result.elapsed_seconds}s · {result.analysis.player_type} · CrickLM local model
                                </p>
                            </div>
                            <Btn variant="outline" onClick={reset} style={{ padding: "8px 18px", fontSize: 12 }}>
                                ← New analysis
                            </Btn>
                        </div>
                        <ResultPanel result={result} />
                    </div>
                )}
            </div>
        </div>
    );
}

// ROOT

export default function App() {
    const [view, setView] = useState<"landing" | "app">("landing");
    const [scrollY, setScrollY] = useState(0);

    useEffect(() => {
        const onScroll = () => setScrollY(window.scrollY);
        window.addEventListener("scroll", onScroll, { passive: true });
        return () => window.removeEventListener("scroll", onScroll);
    }, []);

    const launchApp = () => { setView("app"); window.scrollTo(0, 0); };
    const goBack = () => { setView("landing"); window.scrollTo(0, 0); };

    return (
        <>
            <style>{CSS}</style>
            {view === "app" ? (
                <AnalysisApp onBack={goBack} />
            ) : (
                <>
                    <Navbar scrollY={scrollY} onCTA={launchApp} />
                    <Hero scrollY={scrollY} onCTA={launchApp} />
                    <Pipeline />
                    <Features />
                    <FinalCTA onCTA={launchApp} />
                    <Footer />
                </>
            )}
        </>
    );
}