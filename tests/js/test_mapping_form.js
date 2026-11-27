// Unit tests for diacritical/static/js/mapping_form.js (the Mappings page's Type boxes <-> stored code).
// Run: node tests/js/test_mapping_form.js [mappings.csv]
// With a CSV of the list (columns plain_text,diacritic_text; default $MAPPINGS_CSV or /root/case/mappings-live.csv
// when present) every entry is round-tripped through the boxes and the ones left to Advanced are listed.
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const M = require(path.join(__dirname, '..', '..', 'diacritical', 'static', 'js', 'mapping_form.js'));

let passed = 0;
const failures = [];
function test(name, fn) {
    try {
        fn();
        passed += 1;
    } catch (e) {
        failures.push(name + ': ' + e.message);
    }
}

function boxes(over) {
    return Object.assign(M.emptyBoxes(), over);
}

// Each example both ways: the boxes give the code, and the code gives the boxes back.
const EXAMPLES = [
    ['aks͟haya', { type: 'normal', result: 'aks͟haya' }],
    ['kṛs͟hṇa', { type: 'normal', result: 'kṛs͟hṇa' }],
    ['{{}}{{koḍiāl}}', { type: 'capital', result: 'koḍiāl' }],
    ['{{gŏḍ}}{{god}}', { type: 'normal', result: 'gŏḍ', name: 'god' }],
    ['{{abhis͟heka}}{{abhis͟hek}}', { type: 'normal', result: 'abhis͟heka', name: 'abhis͟hek' }],
    ['{{nāmāvalī (hindi), nāmāvaliḥ (sanskrit)}}', { type: 'choices', choices: ['nāmāvalī (hindi)', 'nāmāvaliḥ (sanskrit)'] }],
    ['{{}}{{kārtik, kārtika}}', { type: 'choices', choices: ['kārtik', 'kārtika'], capital: true }],
    // Choices that give a name when typed with a capital (Shantish's arati), and a comma inside a label.
    ['{{ārati (sanskrit), āratī (hindi, marathi)}}{{āratī}}',
        { type: 'choices', choices: ['ārati (sanskrit)', 'āratī (hindi, marathi)'], name: 'āratī' }],
    ['{{varsha, varshā}}{{varshā}}', { type: 'choices', choices: ['varsha', 'varshā'], name: 'varshā' }],
    ['{{ārati (sanskrit), āratī (hindi, marathi)}}',
        { type: 'choices', choices: ['ārati (sanskrit)', 'āratī (hindi, marathi)'] }],
    ['{{jñāta, jñātā, jñāti}}', { type: 'choices', choices: ['jñāta', 'jñātā', 'jñāti'] }],
    ['{{}}{{parijñānāshrama trayodashī}}', { type: 'capital', result: 'parijñānāshrama trayodashī' }],
];

for (const [code, want] of EXAMPLES) {
    test('boxes -> code ' + code, () => assert.strictEqual(M.boxesToCode(boxes(want)), code));
    test('code -> boxes ' + code, () => assert.deepStrictEqual(M.codeToBoxes(code), boxes(want)));
}

test('boxes are cleaned to the house style', () => {
    assert.strictEqual(M.boxesToCode(boxes({ type: 'normal', result: '  Kṛs̲h̲ṇa ' })), 'kṛs͟hṇa');
    assert.strictEqual(M.boxesToCode(boxes({ type: 'normal', result: ' Gŏḍ', name: 'God ' })), '{{gŏḍ}}{{god}}');
    assert.strictEqual(M.boxesToCode(boxes({ type: 'choices', choices: [' a ', '', 'b  (Hindi)'] })), '{{a, b (hindi)}}');
});

test('a filled Advanced box wins', () => {
    assert.strictEqual(M.boxesToCode(boxes({ type: 'normal', result: 'x', advanced: ' {{varsha, varshā}} {{varshā}}' })),
        '{{varsha, varshā}}{{varshā}}');
    assert.strictEqual(M.boxesToCode(boxes({ type: 'normal', result: 'x', advanced: '   ' })), 'x');
});

test('broken brackets are repaired before reading', () => {
    assert.deepStrictEqual(M.codeToBoxes('{}}{{hebl̃ekar}}'), boxes({ type: 'capital', result: 'hebl̃ekar' }));
    assert.deepStrictEqual(M.codeToBoxes(' {{}} {{Māṇāyī}} '), boxes({ type: 'capital', result: 'māṇāyī' }));
});

// Codes the boxes can't show exactly stay as a code (Advanced), with a reason.
const ADVANCED = [
    ['{{saṅgīta}}{{saṅgeetā, saṅgītā}}', 'several spellings for when'],
    ['{{a, b}}{{c, d}}', 'several spellings for when'],
    ['{{a}}{{b}}{{c}}', 'brackets'],
    ['x{{y}}', 'brackets'],
    ['{{y}}x', 'brackets'],
    ['a, b', 'comma'],
    ['{{lalita,lalitā}}', 'slightly differently'],
    ['{{lalita, , lalitā}}', 'slightly differently'],
    ['{{lalita}}', 'one spelling'],
    ['{{}}', 'nothing'],
    ['{{}}{{}}', 'no spelling'],
    ['{{x}}{{}}', 'empty'],
    ['{{x', 'brackets'],
];
for (const [code, why] of ADVANCED) {
    test('advanced ' + code, () => {
        const b = M.codeToBoxes(code);
        assert.strictEqual(b.type, 'advanced');
        assert.strictEqual(b.advanced, M.cleanResult(code));
        assert.ok(b.why.includes(why), b.why);
        assert.strictEqual(M.boxesToCode(b), M.cleanResult(code));
    });
}

test('a comma inside a label stays in its choice', () => {
    assert.deepStrictEqual(M.splitChoiceLines('ārati (sanskrit)\nāratī (hindi, marathi)'),
        ['ārati (sanskrit)', 'āratī (hindi, marathi)']);
    assert.deepStrictEqual(M.splitChoiceLines('a, b (x, y)'), ['a', 'b (x, y)']);
    assert.strictEqual(M.describeCode('{{ārati (sanskrit), āratī (hindi, marathi)}}{{āratī}}', 'arati'),
        'arati → ārati (sanskrit) or āratī (hindi, marathi), Arati → Āratī');
});

test('the capital spelling box is optional, and ignored when always capital', () => {
    const b = boxes({ type: 'choices', choices: ['a', 'b'], name: '' });
    assert.deepStrictEqual(M.boxProblems(b, true), []);
    assert.strictEqual(M.boxesToCode(b), '{{a, b}}');
    assert.strictEqual(M.boxesToCode(Object.assign(b, { name: 'b' })), '{{a, b}}{{b}}');
    assert.strictEqual(M.boxesToCode(Object.assign(b, { capital: true })), '{{}}{{a, b}}');
    assert.strictEqual(M.boxesToCode(boxes({ type: 'capital', result: 'x', name: 'y' })), '{{}}{{x}}');
    assert.strictEqual(M.boxesToCode(boxes({ type: 'normal', result: 'x', name: '' })), 'x');
    assert.strictEqual(M.boxesToCode(boxes({ type: 'name', result: 'gŏḍ', name: 'god' })), '{{gŏḍ}}{{god}}');
});

test('problems in plain words', () => {
    const p = (over, saving = true) => M.boxProblems(boxes(over), saving);
    assert.deepStrictEqual(p({ type: 'normal', result: 'kṛs͟hṇa' }), []);
    assert.deepStrictEqual(p({ type: 'normal', result: '' }), ['Please type the Result (the word with its marks).']);
    assert.deepStrictEqual(p({ type: 'normal', result: '' }, false), []);
    assert.ok(p({ type: 'normal', result: 'a, b' })[0].includes('choose Choices'));
    assert.ok(p({ type: 'capital', result: '{{x}}' })[0].includes('choose Choices'));
    assert.ok(p({ type: 'normal', result: 'gŏḍ', name: 'god, gods' })[0].includes('capital spelling'));
    assert.deepStrictEqual(p({ type: 'normal', result: 'gŏḍ', name: '' }), []);
    assert.deepStrictEqual(p({ type: 'choices', choices: ['a'] }), ['Choices needs at least two spellings, one per line.']);
    assert.deepStrictEqual(p({ type: 'choices', choices: ['a', '{{b}}'] }), ['Please type the spellings without {{ }}.']);
    assert.deepStrictEqual(p({ type: 'choices', choices: ['a', 'b'] }), []);
    assert.deepStrictEqual(p({ type: 'normal', result: '', advanced: '{{a}}{{b}}' }), []);
    for (const msg of [].concat(p({ type: 'normal' }), p({ type: 'normal', result: 'x', name: 'a,b' }), p({ type: 'choices' }))) {
        assert.ok(!/\{\{\}\}/.test(msg), 'no codes in messages: ' + msg);
    }
});

test('what it will give', () => {
    assert.strictEqual(M.describeCode('{{gŏḍ}}{{god}}', 'god'), 'god → gŏḍ, God → God');
    assert.strictEqual(M.describeCode('kṛs͟hṇa', 'krishna'), 'krishna → kṛs͟hṇa, Krishna → Kṛs͟hṇa');
    assert.strictEqual(M.describeCode('aks͟haya', ''), 'aks͟haya, with a capital: Aks͟haya');
    assert.strictEqual(M.describeCode('{{}}{{koḍiāl}}', 'kodial'), 'kodial → Koḍiāl (always)');
    assert.strictEqual(M.describeCode('{{}}{{koḍiāl}}', ''), 'Koḍiāl (always)');
    assert.strictEqual(M.describeCode('{{nāmāvalī (hindi), nāmāvaliḥ (sanskrit)}}', ''),
        'nāmāvalī (hindi) or nāmāvaliḥ (sanskrit)');
    assert.strictEqual(M.describeCode('{{}}{{kārtik, kārtika}}', 'kartik'), 'kartik → Kārtik or Kārtika (always)');
    assert.strictEqual(M.describeCode('{{varsha, varshā}}{{varshā}}', 'varsha'), 'varsha → varsha or varshā, Varsha → Varshā');
    assert.strictEqual(M.describeCode('{{}}{{parijñānāshrama trayodashī}}', ''), 'Parijñānāshrama Trayodashī (always)');
    assert.strictEqual(M.describeCode('{{a}}{{b}}{{c}}', 'x'), '');
});

test('choices box: one spelling per line, commas start a new line', () => {
    assert.strictEqual(M.cleanChoicesText(' Kārtik, kārtika \n\n  nāmāvalī  (Hindi)\n'), 'kārtik\nkārtika\nnāmāvalī (hindi)');
    assert.deepStrictEqual(M.splitChoiceLines('a\n b \n,c'), ['a', 'b', 'c']);
});

test('the house style, as cleaning.py', () => {
    assert.strictEqual(M.cleanPlain('  Nāmāvalih  '), 'namavalih');
    assert.strictEqual(M.cleanResult('dus\u0332h\u0332kāl̃a'), 'dus\u035fhkāl̃a');
    assert.strictEqual(M.cleanResult('{{kākḍā}}}'), '{{kākḍā}}');
    assert.strictEqual(M.cleanResult('d\u0323'), '\u1e0d');
});

// ---------- the real list ----------

function readCsv(text) {
    const rows = [];
    let row = [], field = '', quoted = false;
    for (let i = 0; i < text.length; i++) {
        const c = text[i];
        if (quoted) {
            if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
            else if (c === '"') quoted = false;
            else field += c;
        } else if (c === '"') quoted = true;
        else if (c === ',') { row.push(field); field = ''; }
        else if (c === '\n' || c === '\r') {
            if (c === '\r' && text[i + 1] === '\n') i++;
            row.push(field); rows.push(row); row = []; field = '';
        } else field += c;
    }
    if (field || row.length) { row.push(field); rows.push(row); }
    return rows;
}

const csvPath = process.argv[2] || process.env.MAPPINGS_CSV || '/root/case/mappings-live.csv';
let listReport = 'real list: not checked (no CSV at ' + csvPath + ')';
if (fs.existsSync(csvPath)) {
    const rows = readCsv(fs.readFileSync(csvPath, 'utf8'));
    const head = rows.shift();
    const pi = head.indexOf('plain_text'), di = head.indexOf('diacritic_text');
    const counts = { normal: 0, 'normal + capital spelling': 0, capital: 0, choices: 0, 'choices + capital spelling': 0,
        'choices + always capital': 0, advanced: 0 };
    const advanced = [];
    let total = 0;
    for (const r of rows) {
        if (r.length < 2) continue;
        total += 1;
        const code = r[di];
        const b = M.codeToBoxes(code);
        test('round trip ' + r[pi], () => assert.strictEqual(M.boxesToCode(b), M.cleanResult(code)));
        if (b.type === 'advanced') {
            counts.advanced += 1;
            advanced.push(`  ${r[pi]} -> ${code}   (${b.why})`);
        } else {
            counts[b.type === 'choices' && b.capital ? 'choices + always capital' : b.name ? b.type + ' + capital spelling' : b.type] += 1;
            // the boxes as they'd be typed (lines in the Choices box) give the same code
            if (b.type === 'choices') {
                test('choices box ' + r[pi], () => assert.strictEqual(
                    M.boxesToCode(Object.assign({}, b, { choices: M.splitChoiceLines(b.choices.join('\n')) })),
                    M.cleanResult(code)));
            }
        }
    }
    listReport = `real list ${csvPath}: ${total} entries, ${total - counts.advanced} round-trip through the boxes, ` +
        `${counts.advanced} left to Advanced\n  by type: ` + JSON.stringify(counts) +
        (advanced.length ? '\n Advanced:\n' + advanced.join('\n') : '');
}

console.log(listReport);
if (failures.length) {
    console.log(`${failures.length} FAILED, ${passed} passed`);
    failures.slice(0, 50).forEach((f) => console.log('  FAIL ' + f));
    process.exit(1);
}
console.log(`ok: ${passed} passed`);
