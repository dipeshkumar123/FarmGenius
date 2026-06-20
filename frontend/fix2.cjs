const fs = require('fs');

let f0 = fs.readFileSync('src/pages/LandingPage.tsx', 'utf8');
f0 = f0.replace(/,\s*ease:\s*(?:'[^']+'|"[^"]+"|\[[\d.,\s]+\])/g, '');
f0 = f0.replace(/ease:\s*(?:'[^']+'|"[^"]+"|\[[\d.,\s]+\]),?\s*/g, '');
fs.writeFileSync('src/pages/LandingPage.tsx', f0);

let f1 = fs.readFileSync('src/pages/LoginPage.tsx', 'utf8');
f1 = f1.replace(/import \{ useState, useRef, useEffect, KeyboardEvent \}/g, "import { useState, useRef, useEffect } from 'react';\nimport type { KeyboardEvent }");
f1 = f1.replace(/Plant,/g, 'Leaf as Plant,'); 
fs.writeFileSync('src/pages/LoginPage.tsx', f1);

console.log("Fixes applied successfully.");
