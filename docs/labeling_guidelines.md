
# Helpdesk Dataset — Labeling Guidelines

## Purpose

This document defines consistent labeling rules for the IT Helpdesk ticket classification dataset.

Each ticket must receive exactly one of the six supported labels.

The primary classification criterion is the user's requested action or reported issue, rather than the presence of specific keywords.

## 1. PASSWORD_RESET

**Definition:** Requests involving password recovery, credential resets, expired passwords, and account lockouts.

**Include:**
- Forgotten passwords.
- Expired credentials.
- Locked accounts requiring an unlock.
- Requests for password reset links.

**Exclude:**
- Missing permissions to access a resource.
- General login instructions without a reported problem.

## 2. ACCESS_ISSUE

**Definition:** Problems involving authentication, authorization, permissions, and access to protected systems.

**Include:**
- VPN authentication problems.
- Access denied errors.
- Missing file or application permissions.
- Requests to provision user access.

**Exclude:**
- Explicit password reset requests.
- General connectivity outages.

## 3. HARDWARE_FAILURE

**Definition:** Reported failures or physical malfunctions involving computing equipment and peripherals.

**Include:**
- Broken monitors or keyboards.
- Laptop power failures.
- Damaged cables.
- Malfunctioning physical peripherals.

**Exclude:**
- Software installation requests.
- Network-wide connectivity problems.

## 4. SOFTWARE_INSTALL

**Definition:** Requests to install, reinstall, deploy, license, or provision software.

**Include:**
- Installing applications.
- Reinstalling corrupted software.
- Software licensing requests.
- Deploying approved software packages.

**Exclude:**
- Requests for user access permissions unrelated to provisioning software.
- Questions about how to use existing applications.

## 5. NETWORK_LATENCY

**Definition:** Reported network performance, stability, and connectivity problems.

**Include:**
- Slow Wi-Fi or Ethernet connections.
- High latency or packet loss.
- Network disconnections.
- Unreachable websites caused by suspected connectivity failures.

**Exclude:**
- VPN authentication failures.
- Missing authorization for specific systems.

**Note:** The current label also includes connectivity failures, despite its narrower name.

## 6. GENERAL_QUERY

**Definition:** Informational IT support questions that do not primarily report a technical failure or request an operational change.

**Include:**
- IT policies and procedures.
- Support contact information.
- Service hours and SLAs.
- General how-to questions.

**Exclude:**
- Explicit requests for software installation.
- Account recovery requests.
- Reported hardware or network failures.

## Ambiguity Resolution

1. Identify the user's main problem or requested action.
2. Prefer the explicit request over assumptions about the technical cause.
3. Do not assign a label based only on keywords.
4. If a ticket explicitly requests password recovery, prefer PASSWORD_RESET over ACCESS_ISSUE.
5. Distinguish network connectivity failures from authorization failures.
6. If the primary intent remains unclear, flag the example for manual review rather than guessing.

## Dataset Quality Rules

- Use realistic English-language helpdesk requests.
- Include short, informal, and detailed ticket descriptions.
- Avoid exact duplicates and near-duplicate paraphrases.
- Maintain consistent category definitions.
- Review ambiguous examples before including them.
- Keep evaluation examples separate from model development and tuning.
- Do not use test-set performance to decide which examples to add to training.
