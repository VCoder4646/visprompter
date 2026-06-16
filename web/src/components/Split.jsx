import React, { useCallback, useEffect, useRef, useState } from "react";

// Two-pane resizable split. direction: "h" (left|right) or "v" (top|bottom).
// children must be exactly two nodes. `initial` is the first pane's % size.
export default function Split({ direction = "h", initial = 40, min = 12, max = 88, children }) {
  const [pct, setPct] = useState(initial);
  const wrapRef = useRef();
  const dragging = useRef(false);
  const horiz = direction === "h";

  const onMove = useCallback((e) => {
    if (!dragging.current || !wrapRef.current) return;
    const r = wrapRef.current.getBoundingClientRect();
    const p = horiz ? ((e.clientX - r.left) / r.width) * 100
                    : ((e.clientY - r.top) / r.height) * 100;
    setPct(Math.min(max, Math.max(min, p)));
  }, [horiz, min, max]);

  useEffect(() => {
    const up = () => { dragging.current = false; document.body.style.userSelect = ""; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", up);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", up); };
  }, [onMove]);

  const start = () => { dragging.current = true; document.body.style.userSelect = "none"; };
  const [a, b] = React.Children.toArray(children);

  return (
    <div ref={wrapRef}
         style={{ display: "flex", flexDirection: horiz ? "row" : "column",
                  width: "100%", height: "100%", minHeight: 0, minWidth: 0 }}>
      <div style={{ flex: `0 0 ${pct}%`, minHeight: 0, minWidth: 0, overflow: "hidden" }}>{a}</div>
      <div onMouseDown={start}
           title="drag to resize"
           style={{
             flex: "0 0 6px", cursor: horiz ? "col-resize" : "row-resize",
             background: "#1c2433",
             borderLeft: horiz ? "1px solid #2a3344" : "none",
             borderTop: horiz ? "none" : "1px solid #2a3344",
           }} />
      <div style={{ flex: "1 1 0", minHeight: 0, minWidth: 0, overflow: "hidden" }}>{b}</div>
    </div>
  );
}
