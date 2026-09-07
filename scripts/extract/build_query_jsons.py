import json
import re
import argparse


def parse_negation(prompt):
    m = re.match(r"^A realistic photo of a scene without an? (.+)$", prompt)
    if not m:
        return None
    return {"object": m.group(1)}


def parse_counting(prompt):
    m = re.match(r"^A realistic photo of a scene with (\d+) (.+)$", prompt)
    if not m:
        return None
    count = int(m.group(1))
    noun_phrase = m.group(2)
    if count == 1:
        object_name = noun_phrase
    else:
        if not noun_phrase.endswith("s"):
            return None
        object_name = noun_phrase[:-1]
    return {"object": object_name, "count": count}


def parse_color_attr(prompt):
    m = re.match(r"^A realistic photo of a scene with an? (.+) and an? (.+)$", prompt)
    if not m:
        return None
    return [m.group(1), m.group(2)]


def parse_position(prompt):
    m = re.match(
        r"^A realistic photo of a scene with an? (.+) on the (left|right|top|bottom) "
        r"and an? (.+) on the (left|right|top|bottom)$",
        prompt
    )
    if not m:
        return None
    obj1, side1, obj2, side2 = m.groups()

    # Relationship phrasing matching test_spatial_relationships.py's own vocabulary:
    # "X is to the right of Y" / "X is to the left of Y" / "X is above Y" / "X is below Y"
    phrasing = {
        "left": f"{obj1} is to the left of {obj2}",
        "right": f"{obj1} is to the right of {obj2}",
        "top": f"{obj1} is above {obj2}",
        "bottom": f"{obj1} is below {obj2}",
    }
    return {"objects": [obj1, obj2], "relationships": [phrasing[side1]]}


PARSERS = {
    "negation": parse_negation,
    "counting": parse_counting,
    "color_attr": parse_color_attr,
    "position": parse_position,
}


def build_query_jsons(dataset_path):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    query_jsons = {}
    failures = []

    for tag, parser in PARSERS.items():
        entries = dataset.get(tag, [])
        query_json = {}
        for entry in entries:
            prompt = entry["prompt"]
            line_number = entry["line_number"]
            parsed = parser(prompt)
            if parsed is None:
                failures.append((tag, line_number, prompt))
                continue
            query_json[str(line_number)] = parsed
        query_jsons[tag] = query_json

    return query_jsons, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="t2i_benchmark_dataset.json")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    query_jsons, failures = build_query_jsons(args.dataset)

    for tag, query_json in query_jsons.items():
        out_path = f"{args.output_dir}/{tag}_specific.json"
        with open(out_path, "w") as f:
            json.dump(query_json, f, indent=4)
        print(f"Wrote {len(query_json)} entries to {out_path}")

    if failures:
        print(f"\n{len(failures)} prompts failed to parse:")
        for tag, line_number, prompt in failures:
            print(f"  [{tag}] line_number={line_number}: {prompt!r}")


if __name__ == "__main__":
    main()