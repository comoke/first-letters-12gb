#!/bin/bash
source "${PC:-$(cd "$(dirname "$0")/.." && pwd)}/env.sh"
$DK run --rm --memory 4g --entrypoint bash fl-claim-a7:final -c 'find / -name v8306lib.py -not -path "/proc/*" 2>/dev/null; find / -name check.py -path "*exclusion*" 2>/dev/null'
