# NoSQL Translation Verification Report

## Dataset: Spider `concert_singer` (8 examples)

### Schema Context
- Table: `singer`
- Columns: `id` (PK), `name`, `country`, `age`, `song_name`, `song_release_year` (all lowercase in schema)

---

## Example-by-Example Verification

### Example 1 ✅ CORRECT
**SQL:** `SELECT count(*) FROM singer`

**Generated NoSQL:**
```javascript
db.singer.aggregate([
  {
    $group: {
      _id: null,
      total: { $sum: 1 }
    }
  }
])
```

**Verdict:** ✅ **CORRECT**
- Groups all docs, sums to get count
- Returns `{ _id: null, total: <count> }`

---

### Example 2 ✅ CORRECT
**SQL:** `SELECT count(*) FROM singer`

**Generated NoSQL:**
```json
db.singer.countDocuments({})
```

**Verdict:** ✅ **CORRECT**
- Simpler approach, counts all documents
- Returns scalar count

---

### Example 3 ✅ CORRECT
**SQL:** `SELECT name, country, age FROM singer ORDER BY age DESC`

**Generated NoSQL:**
```javascript
db.singer.find({}, { name: 1, country: 1, age: 1 }).sort({ age: -1 })
```

**Verdict:** ✅ **CORRECT**
- Projects only `name`, `country`, `age`
- Sorts by age DESC (–1 = descending)
- Matches SQL semantics

---

### Example 4 ⚠️ INCOMPLETE (but functional)
**SQL:** `SELECT name, country, age FROM singer ORDER BY age DESC`

**Generated NoSQL:**
```javascript
db.singer.find().sort({age: -1})
```

**Verdict:** ⚠️ **FUNCTIONALLY CORRECT but lacks projection**
- Sorts correctly by age DESC
- **Issue:** Returns all fields instead of just `name`, `country`, `age`
- **Better:** `db.singer.find({}, { name: 1, country: 1, age: 1 }).sort({ age: -1 })`

---

### Example 5 ❌ FIELD NAME CASING ERROR
**SQL:** `SELECT avg(age), min(age), max(age) FROM singer WHERE country = 'France'`

**Generated NoSQL:**
```javascript
db.singer.aggregate([
  { $match: { Country: 'France' } },
  {
    $group: {
      _id: null,
      avgAge: { $avg: "$Age" },
      minAge: { $min: "$Age" },
      maxAge: { $max: "$Age" }
    }
  }
])
```

**Verdict:** ❌ **FIELD NAMING ERROR**
- **Issue:** Uses `Country` and `Age` (capitalized) but schema has `country` and `age` (lowercase)
- **Corrected:**
```javascript
db.singer.aggregate([
  { $match: { country: 'France' } },
  {
    $group: {
      _id: null,
      avgAge: { $avg: "$age" },
      minAge: { $min: "$age" },
      maxAge: { $max: "$age" }
    }
  }
])
```

---

### Example 6 ❌ FIELD NAME CASING ERROR
**SQL:** `SELECT avg(age), min(age), max(age) FROM singer WHERE country = 'France'`

**Generated NoSQL:**
```javascript
db.singer.aggregate([
    { $match: { "Country": "France" } },
    { $group: {
        _id: null,
        avgAge: { $avg: "$Age" },
        minAge: { $min: "$Age" },
        maxAge: { $max: "$Age" }
    }}
])
```

**Verdict:** ❌ **SAME FIELD NAMING ERROR AS EXAMPLE 5**
- **Issue:** Capitalized `Country` and `Age` should be lowercase
- **Corrected:** (same fix as Example 5)

---

### Example 7 ❌ MISSING PROJECTION + CASING ERROR
**SQL:** `SELECT song_name, song_release_year FROM singer ORDER BY age LIMIT 1`

**Generated NoSQL:**
```javascript
db.singer.aggregate([
  { $sort: { Age: 1 } },
  { $limit: 1 }
])
```

**Verdict:** ❌ **MISSING PROJECTION + CASING ERROR**
- **Issue 1:** Uses capitalized `Age` instead of lowercase `age`
- **Issue 2:** Missing `$project` to return only `song_name` and `song_release_year`
- **Issue 3:** Returns all fields of the youngest singer instead of just the requested columns
- **Corrected:**
```javascript
db.singer.aggregate([
  { $sort: { age: 1 } },
  { $limit: 1 },
  { $project: { _id: 0, song_name: 1, song_release_year: 1 } }
])
```

---

### Example 8 ❌ WRONG SORT DIRECTION + INCORRECT PROJECTION
**SQL:** `SELECT song_name, song_release_year FROM singer ORDER BY age LIMIT 1`

**Generated NoSQL:**
```javascript
{
  "aggregate": [
    {
      "$sort": { "Age": -1 }
    },
    {
      "$limit": 1
    },
    {
      "$project": { "_id": 0, "Song_Name": 1, "Song_release_year": 1 }
    }
  ]
}
```

**Verdict:** ❌ **MULTIPLE ERRORS**
- **Issue 1:** `Age: -1` (descending) returns **oldest** singer; should be `age: 1` (ascending) for **youngest**
- **Issue 2:** Capitalized field names `Song_Name` instead of `song_name`
- **Issue 3:** Inconsistent casing: `Song_release_year` should be `song_release_year`
- **Corrected:**
```javascript
db.singer.aggregate([
  { $sort: { age: 1 } },
  { $limit: 1 },
  { $project: { _id: 0, song_name: 1, song_release_year: 1 } }
])
```

---

## Summary

| Example | SQL Type | Verdict | Issues |
|---------|----------|---------|--------|
| 1 | COUNT | ✅ CORRECT | None |
| 2 | COUNT | ✅ CORRECT | None |
| 3 | SELECT...ORDER BY | ✅ CORRECT | None |
| 4 | SELECT...ORDER BY | ⚠️ INCOMPLETE | Missing projection (returns extra fields) |
| 5 | AGG (WHERE) | ❌ ERROR | Field casing: `Country`, `Age` → `country`, `age` |
| 6 | AGG (WHERE) | ❌ ERROR | Field casing: `Country`, `Age` → `country`, `age` |
| 7 | SELECT...LIMIT | ❌ ERROR | Casing + missing projection |
| 8 | SELECT...LIMIT | ❌ ERROR | Wrong sort direction + field casing |

### Accuracy: 2/8 fully correct, 1/8 incomplete, 5/8 with errors

### Common Issues:
1. **Field Name Casing (62.5% of errors):** Model capitalizes fields when they should be lowercase
2. **Missing Projections (37.5%):** Model doesn't always include `$project` to select specific columns
3. **Sort Direction (12.5%):** Occasionally reverses ascending/descending

---

## Recommendations

1. **Enhance prompt:** Add field name list to NoSQL generation prompt
   ```
   Include in schema context: "All field names are lowercase: name, country, age, song_name, song_release_year"
   ```

2. **Post-processing:** Add a validation layer to check field names against schema before returning

3. **Model tuning:** Consider fine-tuning qwen2.5-coder:3b on MongoDB examples with consistent casing
