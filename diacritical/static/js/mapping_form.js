/**
 * Diacritical Mappings page: the plain boxes (Type, Result, Choices, the capital spelling) <-> the stored code.
 *
 * The stored code (diacritic_text) takes one of these forms (see diacritics.py, apply_mapping):
 *   word                  Normal: follows how it's typed (krishna -> kṛs͟hṇa, Krishna -> Kṛs͟hṇa)
 *   {{}}{{word}}          Always capital: places and names (Koḍiāl)
 *   {{a, b}}              Choices: all shown ({{}}{{a, b}} when always capital)
 *   {{small}}{{capital}}  Normal or Choices with "spelt differently when typed with a capital" filled in
 *                         ({{gŏḍ}}{{god}}: god -> gŏḍ, God -> God). Shantish found a separate "Name" type confusing.
 * Anything else (several capital spellings, odd brackets) is kept as a code in the Advanced box.
 *
 * Pure functions only (no page access), so they also run under node: tests/js/test_mapping_form.js.
 * cleanPlain/cleanResult are the house style, the same as diacritical/cleaning.py on the server.
 */
(function (root) {
    'use strict';

    // ---------- house style (mirrors cleaning.py) ----------

    function cleanPlain(value) {
        return (value || '').normalize('NFKD').replace(/[\u0300-\u036f]/g, '').replace(/\s+/g, ' ').trim().toLowerCase();
    }

    function cleanResult(value) {
        const marks = '[\\u0300-\\u0331\\u0333-\\u036f]*';
        const pair = new RegExp('([\\p{L}\\p{N}_]' + marks + ')\\u0332([\\p{L}\\p{N}_]' + marks + ')\\u0332', 'gu');
        return (value || '').trim().normalize('NFD').replace(pair, '$1\u035f$2').replace(/\u035f+/g, '\u035f')
            .replace(/\s+/g, ' ').replace(/\{+/g, '{{').replace(/\}+/g, '}}').replace(/\}\}\s+\{\{/g, '}}{{')
            .normalize('NFC').toLowerCase();
    }

    // The Choices box: one spelling per line, each in the house style; a comma typed inside a line starts a new one.
    function cleanChoicesText(value) {
        return splitChoiceLines(value).join('\n');
    }

    // A comma between choices; a comma inside a label's brackets, "āratī (hindi, marathi)", stays in its choice
    // (the same rule as CHOICE_COMMA in diacritics.py).
    const CHOICE_COMMA = /,(?![^()]*\))/;

    function splitChoiceLines(value) {
        const lines = [];
        (value || '').split('\n').forEach(function (line) { lines.push.apply(lines, line.split(CHOICE_COMMA)); });
        return lines.map(cleanResult).filter(Boolean);
    }

    // ---------- reading a code ----------

    // {type: normal|capital|name|choices, main, name} or null when the brackets don't make sense.
    function parseCode(code) {
        code = code || '';
        const groups = [];
        code.replace(/\{\{(.*?)\}\}/g, function (m, g) { groups.push(g); return m; });
        if (!groups.length) {
            return /[{}]/.test(code) ? null : { type: 'normal', main: code };
        }
        if (code.replace(/\{\{(.*?)\}\}/g, '').trim() || groups.some(function (g) { return /[{}]/.test(g); })) {
            return null;
        }
        if (groups.length === 1) return { type: 'choices', main: groups[0] };
        if (groups.length === 2) {
            if (!groups[0].trim()) return { type: 'capital', main: groups[1] };
            return { type: 'name', main: groups[0], name: groups[1] };
        }
        return null;
    }

    function splitChoices(group) {
        return (group || '').split(CHOICE_COMMA).map(function (x) { return x.replace(/\s+/g, ' ').trim(); })
            .filter(Boolean);
    }

    function emptyBoxes() {
        return { type: 'normal', result: '', name: '', choices: [], capital: false, advanced: '' };
    }

    // The stored code (house style) for the boxes. A filled Advanced box wins.
    // .name is the optional "spelt differently when typed with a capital" box (Normal and Choices only):
    // filled, it makes the code {{small}}{{capital spelling}} (god -> gŏḍ, but God -> God).
    function boxesToCode(boxes) {
        boxes = boxes || {};
        if ((boxes.advanced || '').trim()) return cleanResult(boxes.advanced);
        const result = (boxes.result || '').trim();
        const name = (boxes.name || '').trim();
        let code;
        if (boxes.type === 'capital') {
            code = '{{}}{{' + result + '}}';
        } else if (boxes.type === 'choices') {
            const inner = (boxes.choices || []).map(function (c) { return (c || '').trim(); }).filter(Boolean).join(', ');
            code = boxes.capital ? '{{}}{{' + inner + '}}' : name ? '{{' + inner + '}}{{' + name + '}}' : '{{' + inner + '}}';
        } else if (boxes.type === 'advanced') {
            code = '';
        } else {
            code = name ? '{{' + result + '}}{{' + name + '}}' : result;   // normal (and the old type 'name')
        }
        return cleanResult(code);
    }

    // The boxes for a stored code. When the boxes can't give back exactly the same code, the result has
    // type 'advanced', the code in .advanced and a plain reason in .why.
    function codeToBoxes(code) {
        const clean = cleanResult(code);
        const p = parseCode(clean);
        const boxes = emptyBoxes();
        let why = '';
        if (p === null) {
            why = 'its {{ }} brackets are not one of the usual forms';
        } else {
            const main = splitChoices(p.main);
            if (p.type === 'normal') {
                if (clean.indexOf(',') >= 0) why = 'it has a comma but no {{ }}';
                else boxes.result = clean;
            } else if (p.type === 'capital') {
                if (main.length === 1) {
                    boxes.type = 'capital';
                    boxes.result = main[0];
                } else if (main.length >= 2) {
                    boxes.type = 'choices';
                    boxes.choices = main;
                    boxes.capital = true;
                } else {
                    why = 'it is always capital but has no spelling';
                }
            } else if (p.type === 'name') {
                const names = splitChoices(p.name);
                if (main.length === 1 && names.length === 1) {
                    boxes.result = main[0];
                    boxes.name = names[0];
                } else if (main.length > 1 && names.length === 1) {
                    boxes.type = 'choices';
                    boxes.choices = main;
                    boxes.name = names[0];
                } else if (names.length > 1) {
                    why = 'it has several spellings for when it is typed with a capital';
                } else {
                    why = 'one of its two {{ }} groups is empty';
                }
            } else if (p.type === 'choices') {
                if (main.length >= 2) {
                    boxes.type = 'choices';
                    boxes.choices = main;
                } else {
                    why = main.length ? 'it has only one spelling inside {{ }}' : 'it has nothing inside {{ }}';
                }
            }
            if (!why && boxesToCode(boxes) !== clean) {
                why = 'the boxes would write it slightly differently (spaces or commas)';
            }
        }
        if (why) {
            const adv = emptyBoxes();
            adv.type = 'advanced';
            adv.advanced = clean;
            adv.why = why;
            return adv;
        }
        return boxes;
    }

    // ---------- what's wrong, in plain words ----------

    function boxProblems(boxes, forSaving) {
        boxes = boxes || {};
        const out = [];
        if ((boxes.advanced || '').trim()) return out;
        const name = (boxes.name || '').trim();
        const nameProblem = 'Please type the capital spelling without commas or {{ }} (one spelling only).';
        if (boxes.type === 'choices') {
            const choices = (boxes.choices || []).filter(function (c) { return (c || '').trim(); });
            if (choices.some(function (c) { return /[{}]/.test(c); })) {
                out.push('Please type the spellings without {{ }}.');
            }
            if (forSaving && choices.length < 2) {
                out.push('Choices needs at least two spellings, one per line.');
            }
            if (!boxes.capital && /[{},]/.test(name)) out.push(nameProblem);
            return out;
        }
        const result = (boxes.result || '').trim();
        if (/[{},]/.test(result)) {
            out.push('Please type the Result without commas or {{ }}; for several spellings choose Choices.');
        }
        if (boxes.type !== 'capital' && /[{},]/.test(name)) out.push(nameProblem);
        if (forSaving && !result) out.push('Please type the Result (the word with its marks).');
        return out;
    }

    // A warning (not blocking) for a code typed in the Advanced box.
    function codeWarning(code) {
        const clean = cleanResult(code);
        if (!clean) return '';
        if (parseCode(clean) === null) {
            return 'The {{ }} brackets in this code are unusual, so the translator may not use it as expected.';
        }
        return '';
    }

    // ---------- what it will give (mirrors apply_mapping in diacritics.py, roughly) ----------

    function capFirst(text) {
        return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
    }

    // Every word separated by a space (or starting a choice) starts with a capital, as diacritics._name_case.
    function nameCase(text) {
        return (text || '').replace(/(^|[\s,{])(\p{L})/gu, function (m, before, letter) {
            return before + letter.toUpperCase();
        });
    }

    // The typed word with a capital: each word of a phrase capitalised.
    function typedCapital(plain) {
        return plain.indexOf(' ') >= 0 ? nameCase(plain) : capFirst(plain);
    }

    // The result as it comes out when the typed word starts with a capital (labels in brackets kept as written).
    function resultCapital(plain, text) {
        const m = /^(.*?)(\s*\(.*\))?$/.exec(text);
        const word = m[1], label = m[2] || '';
        return (plain.indexOf(' ') >= 0 ? nameCase(word) : capFirst(word)) + label;
    }

    function orList(items) {
        if (items.length <= 1) return items.join('');
        return items.slice(0, -1).join(', ') + ' or ' + items[items.length - 1];
    }

    // A short line such as "god → gŏḍ, God → God" for a code and the typed plain word ('' if not typed yet).
    function describeCode(code, plain) {
        const clean = cleanResult(code);
        const p = parseCode(clean);
        plain = cleanPlain(plain);
        if (!clean || p === null) return '';
        const lead = plain ? plain + ' → ' : '';
        const capLead = plain ? typedCapital(plain) + ' → ' : 'with a capital: ';
        const main = p.type === 'normal' ? [clean] : splitChoices(p.main);
        if (!main.length && p.type !== 'name') return '';
        if (p.type === 'capital') {
            return lead + orList(main.map(nameCase)) + ' (always)';
        }
        if (p.type === 'name') {
            const names = splitChoices(p.name);
            if (!main.length || !names.length) return '';
            return lead + orList(main) + ', ' + capLead + orList(names.map(function (n) {
                return resultCapital(plain, n);
            }));
        }
        if (p.type === 'choices') return lead + orList(main);
        return lead + clean + ', ' + capLead + resultCapital(plain, clean);
    }

    const api = {
        cleanPlain: cleanPlain,
        cleanResult: cleanResult,
        cleanChoicesText: cleanChoicesText,
        splitChoiceLines: splitChoiceLines,
        parseCode: parseCode,
        splitChoices: splitChoices,
        emptyBoxes: emptyBoxes,
        boxesToCode: boxesToCode,
        codeToBoxes: codeToBoxes,
        boxProblems: boxProblems,
        codeWarning: codeWarning,
        describeCode: describeCode
    };
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
    } else {
        root.MappingCodes = api;
    }
})(typeof window !== 'undefined' ? window : this);
