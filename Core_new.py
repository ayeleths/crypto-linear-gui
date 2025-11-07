import random
from itertools import combinations
import numpy as np
import galois
from sklearn.metrics import pairwise_distances
from typing import Union, Tuple, Dict, Optional, List, Any, Iterable
from dataclasses import dataclass, asdict
import time
import math
import pprint


# Type alias: accept NumPy arrays or galois FieldArray
ArrayLike = Union[np.ndarray, galois.FieldArray]

# Field GF(2)
GF2 = galois.GF(2)

# ================================
# Utility Functions (GF(2))
# ================================

def find_minimum_hamming_distance(B: ArrayLike) -> int:
    """Compute the minimum Hamming distance between all distinct rows in a binary matrix `B`.

    Notes
    -----
    This preserves the original behavior (minimum pairwise distance between provided rows),
    not the minimum non-zero codeword weight over the span.
    Accepts GF2 FieldArray or ndarray.
    """
    X = np.asarray(B, dtype=int)
    D = (pairwise_distances(X, metric="hamming") * X.shape[1]).astype(int)
    np.fill_diagonal(D, X.shape[1])
    return int(np.min(D))

def gf2_inv(M: ArrayLike) -> galois.FieldArray:
    """Invert a matrix over GF(2) via Gauss–Jordan elimination.

    Works without galois.linalg and raises np.linalg.LinAlgError if singular over GF(2).
    """
    A = GF2(M).copy()
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise np.linalg.LinAlgError("Matrix must be square for inversion")

    I = GF2.Identity(n)
    Aug = GF2(np.hstack((A, I)))

    r = 0
    for c in range(n):
        # Find pivot with a 1 in column c at or below row r
        pivot = None
        for i in range(r, n):
            if Aug[i, c] == 1:
                pivot = i
                break
        if pivot is None:
            raise np.linalg.LinAlgError("Matrix is singular over GF(2)")
        if pivot != r:
            Aug[[r, pivot], :] = Aug[[pivot, r], :]
        # Eliminate all other 1s in column c
        for i in range(n):
            if i != r and Aug[i, c] == 1:
                Aug[i, :] = Aug[i, :] + Aug[r, :]
        r += 1
    inv = Aug[:, n:]
    return inv

def hamming_distance(a: ArrayLike, b: ArrayLike) -> int:
    """Hamming distance between two GF(2) vectors of equal length."""
    return int(np.sum(GF2(a) != GF2(b)))

def is_permutation_matrix(P: ArrayLike) -> bool:
    """Check if P is a permutation matrix (0/1 with exactly one 1 per row/col)."""
    Q = np.asarray(P, dtype=int)
    ones_per_row = np.all(Q.sum(axis=1) == 1)
    ones_per_col = np.all(Q.sum(axis=0) == 1)
    binary = np.all((Q == 0) | (Q == 1))
    return bool(ones_per_row and ones_per_col and binary)

def degree(poly: ArrayLike) -> int:
    coeffs = np.asarray(poly, dtype=int)
    nz = np.nonzero(coeffs)[0]
    return -1 if len(nz) == 0 else len(coeffs) - 1 - nz[0]

def get_binary_vector_from_user(length: int, num_1s: int = -1) -> galois.FieldArray:
    """Interactive prompt for a binary vector of a given length.

    If num_1s >= 0, enforce exactly this number of ones.
    Returns a GF(2) vector (FieldArray) of shape (length,).
    """
    while True:
        prompt = f"Enter a binary vector of length {length}, using only 0 and 1"
        if num_1s != -1:
            prompt += f" with exactly {num_1s} ones"
        prompt += " (e.g. 101001...):\n"

        user_input = input(prompt)
        user_input = user_input.strip().replace(" ", "")

        if len(user_input) != length:
            print(f"Invalid input. Please enter exactly {length} bits.")
            continue
        if any(c not in "01" for c in user_input):
            print("Invalid input. The input vector must consist of 0s and 1s only.")
            continue

        vec = GF2([np.uint8(bit) for bit in user_input])

        if num_1s != -1 and sum(vec.tolist()) != num_1s:
            print(f"Invalid input. The vector must contain exactly {num_1s} ones.")
            continue

        return vec

def _to_list(x):
    # Converts vector-like objects to a list for display or storage (preserves other data types as they are).
    try:
        if hasattr(x, "view") and hasattr(x, "tolist"):
            return x.view(np.ndarray).tolist()
        if hasattr(x, "tolist"):
            return x.tolist()
        return x
    except Exception:
        return x

# ================================
# running classes
# ================================

# ---------- Data class for a single run ----------
@dataclass
class SingleRunReport:
    # Params
    code_parameters: Optional[str]
    n: int
    k: int
    t: int
    error_prob_channel: float
    requested_deliberate_errors: int
    k_tag: int
    is_random_message: bool
    is_retransmission: bool  # True if this attempt is a retransmission (not the first send)

    # Artifacts (GF2 arrays kept as-is; .to_dict() ידע להמיר ל-list אם צריך)
    plaintext: Any           # original_plaintext
    crc_bits: Any            # CRC_bits
    extended_plaintext: Any  # extended_plaintext
    codeword: Any            # c
    deliberate_error_vec: Any
    deliberate_errors: int   # num_e0
    channel_flips: int
    total_errors: int
    received: Any            # r
    m_hat: Any
    recovered_data: Any
    crc_pass: bool
    crc_remainder: Any
    decode_within_radius: bool
    final_success: bool         # is CRC_check == decode_within_radius?

    # Timing
    encoding_time_s: float
    encoding_us_per_plain_bit: float
    encoding_us_per_code_bit: float
    decoding_time_s: float
    decoding_us_per_code_bit: float

    # Build a dict that matches your current schema + timing fields
    def to_dict(self, serializable: bool = False) -> Dict[str, Any]:
        p_key = f"Plaintext ({self.k_tag} bits)"
        crc_key = f"CRC bits ({CRC.generator_degree} bits)"
        ext_key = f"Extended plaintext ({self.k} bits)"
        mhat_key = f"Decoder output m_hat (k= {self.k} bits incl. CRC)"

        def maybe_serialize(x):
            return _to_list(x) if serializable else x

        report = {
            # Inputs / parameters
            "code_parameters": self.code_parameters,
            "n": int(self.n),
            "k": int(self.k),
            "t": int(self.t),
            "error_prob_channel": float(self.error_prob_channel),
            "requested_deliberate_errors": int(self.requested_deliberate_errors),
            "message length": int(self.k_tag),
            "is_random_message": bool(self.is_random_message),
            "Is retransmission?": bool(self.is_retransmission),

            # Artifacts (matching your keys)
            p_key: maybe_serialize(self.plaintext),
            crc_key: maybe_serialize(self.crc_bits),
            ext_key: maybe_serialize(self.extended_plaintext),
            "Codeword c = extended plaintext · G'": maybe_serialize(self.codeword),
            "Total number of deliberate errors ": int(self.deliberate_errors),
            "Number of channel flips": int(self.channel_flips),
            "Total number of errors ": int(self.total_errors),
            "Received r = c ⊕ e_tot": maybe_serialize(self.received),
            mhat_key: maybe_serialize(self.m_hat),
            "Recovered data (strip CRC)": maybe_serialize(self.recovered_data),
            "CRC check on m_hat → pass?": bool(self.crc_pass),
            "CRC remainder": maybe_serialize(self.crc_remainder),

            # Final success flag + transparency
            "CRC success": bool(self.final_success),
            "decode success": bool(self.decode_within_radius),

            # ---- Timing (added) ----
            "encoding_time_s (CRC+concat+mult+e_tot)": float(self.encoding_time_s),
            "encoding_us_per_plain_bit": float(self.encoding_us_per_plain_bit),
            "encoding_us_per_code_bit": float(self.encoding_us_per_code_bit),
            "decoding_time_s (decode+CRC checks)": float(self.decoding_time_s),
            "decoding_us_per_code_bit": float(self.decoding_us_per_code_bit),
        }
        return report




# ---------- ARQ session summary report----------

@dataclass
class ARQSessionReport:
    # --- Outcome ---
    success: bool                          # is the message decoded finally?
    attempts: int                          # total number of attempts in this session
    success_attempt_index: Optional[int]   # 1-based index of the successful attempt, or None
    max_retries: int                       # -1 indicates "unlimited"
    reason: str                            # "CRC+decode pass" / "max_retries_exceeded" / "time_limit_exceeded" / "stopped_without_success"

    # --- Timing (encoding/decoding) ---
    total_encoding_time_s: float           # sum of encoding times across attempts
    #encoding_time_per_plain_bit_us: float
    total_decoding_time_s: float           # sum of decoding times across attempts
    #avg_decoding_time_s: float             # average decoding time per attempt
    #total_dec_time_per_code_bit_us :float
    #avg_dec_time_per_code_bit_us : float

    # --- Channel flips across attempts ---
    total_channel_flips: int               # total flipped bits across all attempts
    total_flips: int
    #avg_channel_flips_per_attempt: float   # average flipped bits per attempt

    # --- CRC statistics across attempts ---
    total_crc_passes: int                  # how many attempts passed CRC           (NEW #5)
    total_crc_pass_worng_decode: int
    total_crc_fail_right_decode: int
    total_final_successes: int             # number of attempts where both CRC and decoding succeeded

    # --- Session / code parameters (for context) ---
    code_parameters: Optional[str]
    n: int
    k: int
    k_tag: int
    error_prob_channel: float
    requested_deliberate_errors: int

    # --- Per-attempt detailed reports (for tables/logs) ---
    attempts_reports: List["SingleRunReport"]

    @staticmethod
    def _compute_aggregates(attempts_reports: List["SingleRunReport"]) -> Dict[str, Any]:
        """Compute session-level aggregates from per-attempt reports."""
        num = max(len(attempts_reports), 1)
        total_dec = sum(float(r.decoding_time_s) for r in attempts_reports)
        total_flips = sum(int(getattr(r, "channel_flips", 0)) for r in attempts_reports)
        total_crc_ok = sum(1 for r in attempts_reports if bool(getattr(r, "crc_pass", False)))
        total_final_successes = sum(1 for r in attempts_reports if bool(getattr(r, "final_success", False)))
        tot_crc_ok_worng_dec = sum(1 for r in attempts_reports if ((bool(getattr(r,"crc_pass",False))==1) and (bool(getattr(r,"decode_within_radius",False))==0)))
        tot_crc_fail_right_decode= sum(1 for r in attempts_reports if ((bool(getattr(r,"crc_pass",False))==0) and (bool(getattr(r,"decode_within_radius",False))==1)))
        tot_err =sum(int(getattr(r, "total_errors", 0))for r in attempts_reports)

        return dict(
            total_decoding_time_s=total_dec,
            total_channel_flips=total_flips,
            total_crc_passes=total_crc_ok,
            total_final_successes=total_final_successes,
            total_crc_ok_worng_dec = tot_crc_ok_worng_dec,
            total_crc_fail_right_decode = tot_crc_fail_right_decode,
            tot_flips = tot_err

        )

    @classmethod
    def from_attempts(
        cls,
        *,
        success: bool,
        success_attempt_index: Optional[int],
        max_retries: int,
        reason: str,
        total_encoding_time_s: float,
        code_parameters: Optional[str],
        n: int,
        k: int,
        k_tag: int,
        error_prob_channel: float,
        requested_deliberate_errors: int,
        attempts_reports: List["SingleRunReport"],
    ) -> "ARQSessionReport":
        """Factory that builds a session report and fills all aggregates."""
        attempts = len(attempts_reports)
        aggr = cls._compute_aggregates(attempts_reports)

        return cls(
            success=success,
            attempts=attempts,
            success_attempt_index=success_attempt_index,
            max_retries=max_retries,
            reason=reason,
            total_encoding_time_s=float(total_encoding_time_s),
            total_decoding_time_s=float(aggr["total_decoding_time_s"]),
            total_channel_flips=int(aggr["total_channel_flips"]),
            total_crc_passes=int(aggr["total_crc_passes"]),
            total_final_successes=int(aggr["total_final_successes"]),
            code_parameters=code_parameters,
            n=int(n), k=int(k), k_tag=int(k_tag),
            error_prob_channel=float(error_prob_channel),
            requested_deliberate_errors=int(requested_deliberate_errors),
            attempts_reports=attempts_reports,
            total_crc_pass_worng_decode= int(aggr["total_crc_ok_worng_dec"]),
            total_crc_fail_right_decode= int(aggr["total_crc_fail_right_decode"]),
            total_flips= int(aggr["tot_flips"])
        )

    def to_dict(self, serializable: bool = False) -> Dict[str, Any]:
        """Return a GUI-friendly dictionary; attempts are expanded via their own to_dict()."""
        return {
            # Outcome
            "success": self.success,
            "attempts": self.attempts,
            "success_attempt_index": self.success_attempt_index,
            "max_retries": self.max_retries,
            "reason": self.reason,

            # Timing
            "total_encoding_time_s": self.total_encoding_time_s,
            "total_decoding_time_s": self.total_decoding_time_s,   # NEW #2
            # Channel flips
            "total_channel_flips": self.total_channel_flips,       # NEW #3
            "total_errors" :self.total_flips,
            # CRC stats
            "total_crc_passes": self.total_crc_passes,             # NEW #5
            "total_final_successes": self.total_final_successes,
            "total_crc_ok_decode_fail": self.total_crc_pass_worng_decode,
            "total_crc_fail_decode_ok": self.total_crc_fail_right_decode,

            # Session params
            "code_parameters": self.code_parameters,
            "n": self.n, "k": self.k, "k_tag": self.k_tag,
            "error_prob_channel": self.error_prob_channel,
            "requested_deliberate_errors": self.requested_deliberate_errors,

            # Attempts (list of dicts, ready for tables)
            "attempts_reports": [
                r.to_dict(serializable=serializable) for r in self.attempts_reports
            ],
        }

# ================================
# Batch (multi-message) simulation
# ================================


@dataclass
class BatchMessageSummary:
    """One message (with ARQ) rolled up for batch stats."""
    attempts: int
    success: bool
    reason: str  # "CRC+decode pass" / "max_retries_exceeded" / "time_limit_exceeded" / "stopped_without_success"
    # correctness across attempts (per transmission)
    crc_pass_and_wrong_decode: int   # false positives per attempts
    crc_fail_but_right_decode: int   # false negatives per attempts
    crc_right_detection: int         # TP + TN per attempts
    # timing
    encoding_time_per_plain_bit_us: float # new_1
    avg_decoding_time_per_code_bit_us: float
    # errors
    avg_channel_flips_per_attempt : float
    avg_total_errors_per_attempt : float

    # BER (Bit Error Rate):
    #---------------------
    '''
    The Bit Error Rate measures the fraction of bits that were received incorrectly.
    It is calculated as the number of bit positions that differ between the
    transmitted (original) and received (decoded) messages, divided by the total
    number of bits in the message.

    Example:
        BER = (number of bit flips) / (total bits)

    Two common types of BER:
      1. Channel BER  – errors introduced by the noisy communication channel
      2. Decoded BER  – residual errors remaining after error-correction decoding

    A lower BER after decoding indicates better code performance.
    '''

    avg_channel_ber_per_transmission : float
    avg_err_ber_per_transmition: float

@dataclass
class BatchSimulationReport:
    """Aggregated statistics over many messages (each with its own ARQ session)."""
    code_parameters: Optional[str]
    n: int
    k: int
    t: int
    k_tag: int
    error_prob_channel: float
    deliberate_errors_requested: int
    num_messages: int
    total_transmissions: int  # sum of attempts over all messages

    # Averages required by user
    #errors
    avg_channel_errors_per_transmission: float
    avg_total_errors_per_transmission: float
    # BER (Bit Error Rate):
    # ---------------------
    '''
    The Bit Error Rate measures the fraction of bits that were received incorrectly.
    It is calculated as the number of bit positions that differ between the
    transmitted (original) and received (decoded) messages, divided by the total
    number of bits in the message.

    Example:
        BER = (number of bit flips) / (total bits)

    Two common types of BER:
      1. Channel BER  – errors introduced by the noisy communication channel
      2. Decoded BER  – residual errors remaining after error-correction decoding

    A lower BER after decoding indicates better code performance.
    '''

    avg_channel_ber_per_transmission: float
    avg_err_ber_per_transmition: float

    #times
    avg_encoding_time_per_plain_bit_us: float
    avg_decoding_time_per_code_bit_us: float

    eventually_succeeded_pct: float                 # % messages decoded correctly in the end
    avg_attempts_per_message: float

    false_positive_pct: float                       # per-transmission
    false_negative_pct: float                       # per-transmission
    crc_right_decodability_detection_pct: float     # (TP+TN)/attempts

    # Optional: keep per-message rollups if GUI wants a table
    per_message: List[BatchMessageSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Code Parameters": self.code_parameters,
            "n": self.n, "k": self.k, "t": self.t, "k_tag": self.k_tag,
            "Channel Error Probability (p)": self.error_prob_channel,
            "Deliberate Errors (Encoder Noise)": self.deliberate_errors_requested,
            "Number of Messages": self.num_messages,
            "Total Transmissions": self.total_transmissions,

            "Average Channel Errors per Transmission": self.avg_channel_errors_per_transmission,
            "Average Total Errors per Transmission": self.avg_total_errors_per_transmission,
            "Average BER channel Error per Transmission" : self.avg_channel_ber_per_transmission,
            "Average BER Error per Transmission": self.avg_err_ber_per_transmition,

            "Average Attempts per Message": self.avg_attempts_per_message,
            "Decoder Success Rate (%)": self.eventually_succeeded_pct,
            "False Positives (%)": self.false_positive_pct,
            "False Negatives (%)": self.false_negative_pct,
            "Eventually Succeeded (%)": self.eventually_succeeded_pct,
            "CRC Correct Decodability Detection (%)": self.crc_right_decodability_detection_pct,
            "Average Encoding Time per plaintext bit (us)": self.avg_encoding_time_per_plain_bit_us,
            "Average Decoding Time per code bit (us)": self.avg_decoding_time_per_code_bit_us,

            "per_message": [vars(x) for x in self.per_message],
        }

    def metrics_for_plots(self) -> dict:
        """
        Return a simple dict with the key metrics the GUI needs for plots/Excel.
        Contains only numbers + basic parameters.
        """
        return {
            "code_parameters": self.code_parameters,
            "n": self.n,
            "k": self.k,
            "t": self.t,
            "k_tag": self.k_tag,
            "error_prob_channel": self.error_prob_channel,
            "deliberate_errors_requested": self.deliberate_errors_requested,

            "avg_channel_errors_per_transmission" : self.avg_channel_errors_per_transmission,
            "decoder_success_pct": self.eventually_succeeded_pct,
            "crc_detection_pct": self.crc_right_decodability_detection_pct,
            "false_positive_pct": self.false_positive_pct,
            "false_negative_pct": self.false_negative_pct,
            "Average transmissions per Message": self.avg_attempts_per_message,
            "Average BER channel Error per Transmission": self.avg_channel_ber_per_transmission,
            "Average BER Error per Transmission": self.avg_err_ber_per_transmition,

        }

def _bool_equal(a: Any, b: Any) -> bool:
    # Safe equality for GF(2)/numpy vectors
    import numpy as np
    return bool(np.array_equal(np.asarray(a, dtype=int), np.asarray(b, dtype=int)))

def simulate_messages_batch(
    *,
    code,
    error_prob_channel: float,
    errors_num: int,
    is_random_message: bool,
    decoder,
    k_tag: int,
    num_messages: int,
    plaintexts: Optional[Iterable[ArrayLike]] = None,  # or supply explicit messages
    max_retries_per_msg: Optional[int] = None,
    max_duration_per_msg_s: Optional[float] = None,
    keep_deliberate_errors: bool = True
) -> BatchSimulationReport:
    """
    Run a batch of messages. Each message is transmitted with Stop-and-Wait ARQ
    (using arq_transmit_until_success_for_message), collecting the requested statistics.

    Returns a BatchSimulationReport suitable for GUI consumption (.to_dict()).
    """

    per_msg: List[BatchMessageSummary] = []

    total_attempts = 0
    total_channel_flips = 0
    total_errors_all = 0
    total_success_messages = 0

    # Per-transmission confusion tallies across the whole batch
    fp_count = 0  # CRC passed but decoded message != original (false positive)
    fn_count = 0  # CRC failed but decoded message == original (false negative)
    right_detection_count = 0 #  of CRC

    total_enc_time_per_bit = 0.0    # per plain bit
    total_dec_time_per_bit = 0.0    # per code bit

    plaintexts_list = None
    if plaintexts is not None:
        plaintexts_list = list(plaintexts)
        if len(plaintexts_list) != num_messages:
            raise ValueError(
                f"Expected {num_messages} plaintexts, got {len(plaintexts_list)}"
            )
    for m in range(num_messages):
        # One message (with ARQ until success/limit)
        plain = None
        use_random = is_random_message

        if plaintexts_list is not None:
            plain = plaintexts_list[m]
            use_random = False

        session = arq_transmit_until_success_for_message(
            code=code,
            error_prob_channel=error_prob_channel,
            errors_num=errors_num,
            is_random_message=use_random,
            plaintext= plain,
            decoder=decoder,
            k_tag=k_tag,
            max_retries=max_retries_per_msg,
            max_duration_s=max_duration_per_msg_s,
            keep_deliberate_errors=keep_deliberate_errors
        )

        # rollups for this message
        attempts = len(session.attempts_reports)
        total_attempts += attempts
        
        #times
        enc_time_per_bit_us = (float((session.total_encoding_time_s * 1e6)/session.k))
        total_enc_time_per_bit += enc_time_per_bit_us
        dec_time_per_bit_us =float(((session.total_decoding_time_s*1e6)/(attempts * session.n)))
        total_dec_time_per_bit += dec_time_per_bit_us


        # errors
        total_channel_flips += session.total_channel_flips
        total_errors_all += session.total_flips
        avg_channel_flips_per_m = (float(session.total_channel_flips)/attempts)
        avg_errors_per_m = float(session.total_flips/attempts)

        avg_channel_ber_per_m = float ((session.total_channel_flips/ (attempts * session.k))/attempts)
        avg_err_ber_per_m = float((session.total_flips/(attempts * session.k))/attempts)

        # Confusion matrix per transmission within this message
        fp_count += session.total_crc_pass_worng_decode
        fn_count += session.total_crc_fail_right_decode
        right_detection_count += session.total_final_successes

        per_msg.append(
            BatchMessageSummary(
                attempts=attempts,
                success=bool(session.success),
                reason= str(session.reason),
                crc_pass_and_wrong_decode=int(session.total_crc_pass_worng_decode),
                crc_fail_but_right_decode=int(session.total_crc_fail_right_decode),
                crc_right_detection=int(session.total_final_successes),
                encoding_time_per_plain_bit_us=float(enc_time_per_bit_us),
                avg_decoding_time_per_code_bit_us=float(dec_time_per_bit_us),
                avg_channel_ber_per_transmission=float(avg_channel_ber_per_m),
                avg_err_ber_per_transmition=float(avg_err_ber_per_m),
                avg_channel_flips_per_attempt= float(avg_channel_flips_per_m),
                avg_total_errors_per_attempt =float(avg_errors_per_m)
            )

        )

        if session.success:
            total_success_messages += 1

    # ---- Aggregation across messages ----
    total_transmissions = max(total_attempts, 1)
    avg_channel_errs_per_tx = total_channel_flips / total_transmissions
    avg_total_errs_per_tx = total_errors_all / total_transmissions
    avg_channel_ber_all = float((total_channel_flips / (total_transmissions * code.k)) / total_transmissions)
    avg_err_ber_per_all = float((total_errors_all / (total_transmissions * code.k)) / total_transmissions)

    avg_attempts_per_msg = total_transmissions / max(num_messages, 1)

    # Confusion-based percentages are per-transmission
    false_positive_pct = 100.0 * (fp_count / max(total_transmissions, 1))
    false_negative_pct = 100.0 * (fn_count / max(total_transmissions, 1))
    crc_right_detection_pct = 100.0 * (right_detection_count / max(total_transmissions, 1))

    eventually_succeeded_pct = 100.0 * (total_success_messages / max(num_messages, 1))

    # Metadata from code
    code_params = getattr(code, "name", None)

    return BatchSimulationReport(
        code_parameters=code_params,
        n=int(code.n),
        k=int(code.k),
        t=int(code.max_errors_num),
        k_tag=int(k_tag),
        error_prob_channel=float(error_prob_channel),
        deliberate_errors_requested=int(errors_num),
        num_messages=max(num_messages, 1),
        total_transmissions=int(total_transmissions),

        avg_channel_errors_per_transmission=float(avg_channel_errs_per_tx),
        avg_total_errors_per_transmission=float(avg_total_errs_per_tx),
        avg_err_ber_per_transmition= float(avg_err_ber_per_all),
        avg_channel_ber_per_transmission= float(avg_channel_ber_all),

        avg_attempts_per_message=float(avg_attempts_per_msg),
        false_positive_pct=float(false_positive_pct),
        false_negative_pct=float(false_negative_pct),
        eventually_succeeded_pct=float(eventually_succeeded_pct),
        crc_right_decodability_detection_pct=float(crc_right_detection_pct),
        avg_encoding_time_per_plain_bit_us=float(total_enc_time_per_bit/max(num_messages, 1)),
        avg_decoding_time_per_code_bit_us=float(total_dec_time_per_bit/max(num_messages, 1)),
        per_message=per_msg
    )





# ---------- Updated function: returns SingleRunReport ----------
def run_single_example_report(code, error_prob_channel: float, errors_num: int,
                              is_random_message: bool, decoder, k_tag: int,
                              plaintext: Optional[galois.FieldArray]= None ,) -> SingleRunReport:
    """
    Same flow as before, but returns a SingleRunReport object.
    Measures encoding time exactly between your 'start/end of encoding' comments,
    and decoding time between your 'start/end of decoding' comments.
    """
    # --------- input validation ----------
    if not (0.0 <= error_prob_channel <= 1.0):
        raise ValueError("error_prob_channel must be in [0, 1].")
    if k_tag != code.k - CRC.generator_degree:
        raise ValueError(f"k_tag must equal code.k - {CRC.generator_degree} (got k_tag={k_tag}, code.k={code.k}).")
    if errors_num < 0:
        raise ValueError("errors_num must be >= 0.")
    if errors_num > code.max_errors_num:
        raise ValueError(
            f"Requested errors ({errors_num}) exceed max correctable t={code.max_errors_num}."
        )

    # --------- sender: pick plaintext ----------
    if plaintext is None:
        original_plaintext = pick_message(k_tag, is_random_message)  # length k_tag
    else:
        original_plaintext =plaintext


    # ============ ENCODING:  ============
    t_enc0_ns = time.perf_counter_ns()

    CRC_bits = CRC.remainder(original_plaintext, shifted=True)                 # CRC calc
    extended_plaintext = GF2(np.hstack((original_plaintext, CRC_bits)))        # concat
    c = extended_plaintext @ code.G_tag                                        # encode (mult)
    t_enc1_ns = time.perf_counter_ns()

    # --------- error injection ----------
    e0 = create_error_vector_checked(code.n, errors_num, code.max_errors_num)  # deliberate errors
    noisy = apply_channel_noise(GF2(np.zeros(code.n, dtype=int)), error_prob_channel, None)  # channel noise
    e_tot = noisy + e0                                                         # total error vector
    r = c + e_tot                                                              # received


    # Radius check (independent of decode timing)
    decode_success = bool(is_within_decoding_radius(e_tot, code.max_errors_num))

    # ============ DECODING ============
    t_dec0_ns = time.perf_counter_ns()

    m_hat = decoder.decode(r)                  # decode (always, per your current code)
    recovered_data = m_hat[:k_tag]             # strip CRC
    crc_rem = CRC.remainder(m_hat, shifted=False)
    crc_pass = bool(np.all(crc_rem == 0))
    final_success = (decode_success == crc_pass)

    t_dec1_ns = time.perf_counter_ns()

    # --------- metrics ----------
    channel_flips = int(np.sum(noisy.view(np.ndarray)))
    num_e_tot = int(np.sum(e_tot.view(np.ndarray)))
    num_e0 = int(np.sum(e0.view(np.ndarray)))

    # --------- timing computations ----------
    enc_time_s = (t_enc1_ns - t_enc0_ns) / 1e9
    dec_time_s = (t_dec1_ns - t_dec0_ns) / 1e9


    enc_us_per_plain_bit = (enc_time_s * 1e6) / code.k      # convert to  µs/bit
    enc_us_per_code_bit  = (enc_time_s * 1e6) /code.n       # convert to  µs/bit
    dec_us_per_code_bit  = (dec_time_s * 1e6) / code.n      # convert to  µs/bit

    # --------- build and return object ----------
    return SingleRunReport(
        code_parameters=getattr(code, "name", None),
        n=int(code.n), k=int(code.k), t=int(code.max_errors_num),
        error_prob_channel=float(error_prob_channel),
        requested_deliberate_errors=int(errors_num),
        k_tag=int(k_tag),
        is_random_message=bool(is_random_message),
        is_retransmission=False,

        plaintext=original_plaintext,
        crc_bits=CRC_bits,
        extended_plaintext=extended_plaintext,
        codeword=c,
        deliberate_error_vec=e0,
        deliberate_errors=num_e0,
        channel_flips=channel_flips,
        total_errors=num_e_tot,
        received=r,
        m_hat=m_hat,
        recovered_data=recovered_data,
        crc_pass=crc_pass,
        crc_remainder=crc_rem,
        decode_within_radius=decode_success,
        final_success=final_success,

        encoding_time_s=enc_time_s,
        encoding_us_per_plain_bit=enc_us_per_plain_bit,
        encoding_us_per_code_bit=enc_us_per_code_bit,
        decoding_time_s=dec_time_s,
        decoding_us_per_code_bit=dec_us_per_code_bit,
    )



def arq_transmit_until_success_for_message(
    code,
    error_prob_channel: float,
    errors_num: int,
    is_random_message: bool,
    decoder,
    k_tag: int,
    plaintext: Optional[galois.FieldArray]= None , #
    max_retries: Optional[int] = None,      # None => unlimited retransmissions
    max_duration_s: Optional[float] = None, # None => no wall-clock limit
    keep_deliberate_errors: bool = True     # Keep deliberate errors fixed across retransmissions (only channel noise changes)
) -> ARQSessionReport:
    """
    Stop-and-Wait ARQ: retransmits the message until successful decoding or limits are reached.
    If max_retries=None → allows unlimited retransmissions.
    Optionally, specify max_duration_s to prevent infinite loops.
    """
    t_start = time.time()

    # ---------- First transmission (reference if encode_once=True) ----------

    first_attempt: SingleRunReport = run_single_example_report(
        code=code,
        error_prob_channel=error_prob_channel,
        errors_num=errors_num,
        is_random_message=is_random_message,
        decoder=decoder,
        k_tag=k_tag,
        plaintext= plaintext
    )

    attempts_reports: List[SingleRunReport] = [first_attempt]
    total_enc = float(first_attempt.encoding_time_s)

    # Immediate success
    if first_attempt.crc_pass:
        return ARQSessionReport.from_attempts(
            success= bool(first_attempt.crc_pass and  first_attempt.decode_within_radius),
            success_attempt_index=1,
            max_retries=max_retries if max_retries is not None else -1,
            reason="CRC+decode pass" if first_attempt.decode_within_radius is True else "CRC pass but decoded message is worng" ,
            total_encoding_time_s=total_enc,
            code_parameters=first_attempt.code_parameters,
            n=first_attempt.n, k=first_attempt.k, k_tag=first_attempt.k_tag,
            error_prob_channel=error_prob_channel,
            requested_deliberate_errors=errors_num,
            attempts_reports=attempts_reports
        )

    # ---------- Fixed artifacts for retransmissions (same message) ----------
    fixed_plain    = first_attempt.plaintext
    fixed_crc_bits = first_attempt.crc_bits
    fixed_ext      = first_attempt.extended_plaintext
    fixed_c        = first_attempt.codeword
    fixed_e0_count = first_attempt.deliberate_errors  # informational only
    fixed_e0_vec = first_attempt.deliberate_error_vec

    attempt_idx = 1
    while True:
        attempt_idx += 1

        # Stop by retries
        if (max_retries is not None) and (attempt_idx > max_retries):
            break

        # Stop by wall-clock
        if (max_duration_s is not None) and ((time.time() - t_start) > max_duration_s):
            break


        # -------- Retransmit without re-encoding: new channel noise only --------
        noisy = apply_channel_noise(GF2(np.zeros(code.n, dtype=int)), error_prob_channel, None)
        channel_flips = int(np.sum(noisy.view(np.ndarray)))

        # combine with deliberate (kept fixed) only if you choose to vary it
        if keep_deliberate_errors and fixed_e0_count > 0:
            e_tot = noisy + fixed_e0_vec
            deliberate_errors_used = fixed_e0_count
        else:
            e0 = create_error_vector_checked(code.n, errors_num, code.max_errors_num)
            e_tot = noisy + e0
            deliberate_errors_used = int(errors_num)

        r = fixed_c + e_tot

        # timed decoding path
        t0 = time.perf_counter_ns()
        m_hat = decoder.decode(r)
        recovered_data = m_hat[:k_tag]
        crc_rem = CRC.remainder(m_hat, shifted=False)
        crc_pass = bool(np.all(crc_rem == 0))
        decode_success = bool(is_within_decoding_radius(e_tot, code.max_errors_num))
        final_success = bool(decode_success == crc_pass)
        t1 = time.perf_counter_ns()

        dec_time_s = (t1 - t0) / 1e9

        rep_attempt = SingleRunReport(
            code_parameters=getattr(code, "name", None),
            n=int(code.n), k=int(code.k), t=int(code.max_errors_num),
            error_prob_channel=float(error_prob_channel),
            requested_deliberate_errors=int(errors_num),
            k_tag=int(k_tag),
            is_random_message=False,
            is_retransmission=True,

            plaintext=fixed_plain,
            crc_bits=fixed_crc_bits,
            extended_plaintext=fixed_ext,
            codeword=fixed_c,
            deliberate_error_vec= fixed_e0_vec,
            deliberate_errors=deliberate_errors_used,
            channel_flips=channel_flips,
            total_errors=int(np.sum(e_tot.view(np.ndarray))),
            received=r,
            m_hat=m_hat,
            recovered_data=recovered_data,
            crc_pass=crc_pass,
            crc_remainder=crc_rem,
            decode_within_radius=decode_success,
            final_success=final_success,

            encoding_time_s=0.0,
            encoding_us_per_plain_bit=0.0,
            encoding_us_per_code_bit=0.0,
            decoding_time_s=dec_time_s,
            decoding_us_per_code_bit=(dec_time_s * 1e6) / int(code.n),
        )

        attempts_reports.append(rep_attempt)

        if crc_pass:
            return ARQSessionReport.from_attempts(
                success=bool(rep_attempt.crc_pass and  rep_attempt.decode_within_radius),
                success_attempt_index=attempt_idx,
                max_retries=max_retries if max_retries is not None else -1,
                reason="CRC+decode pass" if rep_attempt.decode_within_radius is True else "CRC pass but decoded message is worng" ,
                total_encoding_time_s=total_enc,
                code_parameters=rep_attempt.code_parameters,
                n=rep_attempt.n, k=rep_attempt.k, k_tag=rep_attempt.k_tag,
                error_prob_channel=error_prob_channel,
                requested_deliberate_errors=errors_num,
                attempts_reports=attempts_reports
            )

    # ---------- Finished loop without success ----------
    if (max_retries is not None) and (attempt_idx > max_retries):
        reason = "max_retries_exceeded"
    elif (max_duration_s is not None) and ((time.time() - t_start) > max_duration_s):
        reason = "time_limit_exceeded"
    else:
        reason = "stopped_without_success"

    return ARQSessionReport.from_attempts(
        success=False,
        success_attempt_index=None,
        max_retries=max_retries if max_retries is not None else -1,
        reason=reason,
        total_encoding_time_s=total_enc,
        code_parameters=attempts_reports[0].code_parameters if attempts_reports else None,
        n=attempts_reports[0].n if attempts_reports else 0,
        k=attempts_reports[0].k if attempts_reports else 0,
        k_tag=attempts_reports[0].k_tag if attempts_reports else 0,
        error_prob_channel=error_prob_channel,
        requested_deliberate_errors=errors_num,
        attempts_reports=attempts_reports
    )


def print_report_dict(report: dict):
    """
    Nicely print all keys and values from a result dictionary.
    Handles numpy / GF2 arrays and long vectors gracefully.
    """
    import numpy as np

    print("\n" + "="*60)
    print("📊  RUN REPORT SUMMARY")
    print("="*60)

    for key, value in report.items():
        # Handle arrays (e.g., GF(2) vectors)
        if hasattr(value, "shape") or isinstance(value, (list, np.ndarray)):
            arr = np.array(value).astype(int).tolist()
            # Print compactly for long vectors
            # if len(arr) > 20:
            #     arr_str = f"{arr[:10]} ... {arr[-10:]} (len={len(arr)})"
            arr_str = str(arr)
            print(f"\n🔹 {key}:\n{arr_str}")
        else:
            print(f"\n🔹 {key}: {value}")

    print("\n" + "="*60)
    print("✅  End of report")
    print("="*60 + "\n")


def sweep_simulation_grid(
    *,
    code,
    decoder,
    k_tag: int,
    error_probs: Iterable[float],
    deliberate_errors_values: Iterable[int],
    num_messages: int,
    is_random_message: bool = True,
    max_retries_per_msg: Optional[int] = None,
    max_duration_per_msg_s: Optional[float] = None,
    plaintexts: Optional[Iterable[ArrayLike]] = None,
) -> list[dict]:
    """
    Run simulate_messages_batch for each combination of (error_prob_channel, deliberate_errors_requested).
    Returns a single flat list of dicts, each containing all metrics needed for plotting.
    """
    results = []
    for errors_num in deliberate_errors_values:
        for p in error_probs:

            batch = simulate_messages_batch(
                code=code,
                decoder=decoder,
                k_tag=k_tag,
                error_prob_channel=p,
                errors_num=errors_num,
                plaintexts= plaintexts,
                num_messages=num_messages,
                is_random_message=is_random_message,
                max_retries_per_msg=max_retries_per_msg,
                max_duration_per_msg_s=max_duration_per_msg_s,
            )
            metrics = batch.metrics_for_plots()
            results.append(metrics)
    return results

def test_sweep_simulation_grid() -> None:
    """
    Simple console test for sweep_simulation_grid().
    Uses a small Hamming code so the run will be fast.
    Prints a few key metrics for each (p, deliberate_errors) pair.
    """

    # 1) Build a small test code (Hamming [7,4,3])
    A, S, P = LinearCode_dict["Golay(n=23,k=12,d=7)"]
    code = LinearCode(A, S, P, name="Golay(n=23,k=12,d=7)")
    decoder = Decoder(code)

    # 2) Message length without CRC
    k_tag = code.k - CRC.generator_degree
    if k_tag <= 0:
        raise ValueError(
            f"Message is too short to contain CRC bits: k={code.k}, CRC length={CRC.generator_degree}"
        )

    # 3) Define sweep parameters
    error_probs = [0.01, 0.05, 0.1]          # channel error probabilities to test
    deliberate_errors_values = [0, 1, 2, 3]     # number of deliberate errors at the encoder
    num_messages = 50                        # messages per grid point

    print("\n============================================")
    print("🚀 Running test_sweep_simulation_grid()")
    print("Code:", code.name)
    print("n =", code.n, "k =", code.k, "k_tag =", k_tag)
    print("error_probs =", error_probs)
    print("deliberate_errors_values =", deliberate_errors_values)
    print("num_messages per grid point =", num_messages)
    print("============================================\n")

    # 4) Run the sweep
    grid_results = sweep_simulation_grid(
        code=code,
        decoder=decoder,
        k_tag=k_tag,
        error_probs=error_probs,
        deliberate_errors_values=deliberate_errors_values,
        num_messages=num_messages,
        is_random_message=True,
        max_retries_per_msg=None,
        max_duration_per_msg_s=None,
        plaintexts=None,
    )

    # 5) Pretty-print results
    print(f"Total grid points: {len(grid_results)}\n")

    for entry in grid_results:
        p = entry["error_prob_channel"]
        de = entry["deliberate_errors_requested"]
        dec_succ = entry["decoder_success_pct"]
        fp = entry["false_positive_pct"]
        fn = entry["false_negative_pct"]
        crc_det = entry["crc_detection_pct"]

        print("--------------------------------------------")
        print(f"Channel p = {p:.3f}, deliberate errors = {de}")
        print(f"  Decoder Success Rate (%)        : {dec_succ:.2f}")
        print(f"  CRC Correct Decodability (%)    : {crc_det:.2f}")
        print(f"  False Positives (%)             : {fp:.2f}")
        print(f"  False Negatives (%)             : {fn:.2f}")
    print("--------------------------------------------")
    print("✅ sweep_simulation_grid() test finished.\n")



# ================================
# Errors injection
# ================================
def create_error_vector_checked(length: int,
                                requested_errors: int,
                                max_allowed_errors: int) -> galois.FieldArray:
    """
    Create an error vector with exactly `requested_errors` ones (positions chosen uniformly).
    Raises ValueError if requested_errors > max_allowed_errors or requested_errors > length.
    Returns a GF2 field array of shape (length,).
    """
    if requested_errors < 0:
        raise ValueError("requested_errors must be >= 0")
    if requested_errors > max_allowed_errors:
        raise ValueError(f"Requested errors ({requested_errors}) exceed maximum allowed ({max_allowed_errors}).")
    if requested_errors > length:
        raise ValueError(f"Requested errors ({requested_errors}) exceed vector length ({length}).")

    err = GF2.Zeros(length)
    if requested_errors > 0:
        ones_positions = np.random.choice(length, size=requested_errors, replace=False)
        err[ones_positions] = 1
    return err
def apply_channel_noise(error_vector: galois.FieldArray,
                        bit_error_prob: float,
                        rng: Optional[np.random.Generator] = None) -> galois.FieldArray:
    """
    Given an existing GF2 error_vector, flip each bit independently with probability bit_error_prob.
    Returns new GF2 error vector (initial deliberate errors XOR channel flips).
    """
    if not (0.0 <= bit_error_prob <= 1.0):
        raise ValueError("bit_error_prob must be in [0, 1].")

    length = int(error_vector.size)
    # use provided RNG or default numpy
    if rng is None:
        flips = (np.random.rand(length) < bit_error_prob).astype(np.uint8)
    else:
        flips = int(rng.random(length) < bit_error_prob).astype(np.uint8)

    flips_gf2 = GF2(flips)
    # XOR deliberate errors with flips (in GF2 addition == XOR)
    return error_vector + flips_gf2

def error_weight(e) -> int:
    return int(np.sum(e.view(np.ndarray)))

def is_within_decoding_radius(e, t: int) -> bool:
    return error_weight(e) <= t

# ================================
# Simulation helpers (stateless)
# ================================

def pick_message(k: int, is_random: bool) -> galois.FieldArray:
    return GF2.Random(k) if is_random else get_binary_vector_from_user(k)


# ================================
# Core Classes
# ================================

class LinearCode:
    """Linear code with public generator G_tag = S @ G @ P over GF(2).

    Holds only algebraic/code parameters. No runtime state.
    Attributes:
        G, H : systematic generator and parity-check matrices over GF(2)
        S, P : scrambler (left) and permutation (right)
        G_tag : public generator
        k, n : dimensions
        d_min : pairwise-row minimum distance (original definition)
        max_errors_num (t) : floor((d_min - 1) / 2)
    """

    def __init__(self, A:ArrayLike, S:ArrayLike, P:ArrayLike, name:str):
                # Ensure all inputs are in GF(2)
        A = GF2(A)
        S = GF2(S)
        P = GF2(P)
        # Systematic G = [I | A], H = [A^T | I]
        self.name = name
        self.G = GF2(np.hstack((GF2.Identity(A.shape[0]), A)))
        self.H = GF2(np.hstack((A.T, GF2.Identity(A.shape[1]))))

        # Public key components (GF2-native)
        self.S = S
        self.P = P


        # Basic validations (fail fast with clear messages)
        try:
            _ = gf2_inv(self.S)
        except np.linalg.LinAlgError:
            raise ValueError("S is not invertible over GF(2).")
        if not is_permutation_matrix(self.P):
            raise ValueError("P must be a permutation matrix over GF(2).")

        self.G_tag = self.S @ self.G @ self.P

        # Dimensions and distance parameters
        self.k, self.n = self.G_tag.shape
        self.d_min = find_minimum_hamming_distance(self.G_tag)
        self.max_errors_num = (self.d_min - 1) // 2


        print(f"Code initialized: [n={self.n}, k={self.k}, d_min={self.d_min}]")

class CRC:
    """CRC with generator g(x) = x^3 + x + 1 over GF(2).

    Encoding convention:
    - For a data vector d of length L, the transmitted message is [d | r],
      where r is the 3-bit remainder of x^3 * d(x) modulo g(x).
    - For a fixed-length vector v of length N (N>=3), `set_suffix(v)` replaces
      the last 3 bits with the CRC of the first N-3 bits.
    - `check(v)` returns (ok, remainder) where ok is True iff the full vector
      divides by g(x) with zero remainder (i.e., CRC passes).
    """
    generator = GF2([1, 0, 1, 1])  # CRC= X^3+X+1
    generator_degree = degree(generator)

    @staticmethod
    def remainder(bits: ArrayLike, shifted: bool = False) -> galois.FieldArray:
        """
        Compute the polynomial remainder over GF(2).

        If shifted=False (default):
            Return r(x) = bits(x) mod g(x).
        If shifted=True:
            Return r(x) = (x^R * bits(x)) mod g(x)  (CRC of data).

        Always returns exactly R bits (MSB-first), zero-padded if needed.
        """
        R = CRC.generator_degree
        f = galois.Poly(GF2(bits), field=GF2, order="desc")
        g = galois.Poly(CRC.generator, field=GF2, order="desc")
        if shifted:
            # multiply f(x) by x^R (equivalent to appending R zeros at the end)
            f = galois.Poly(np.concatenate((f.coeffs, np.zeros(R, dtype=int))), field=GF2, order="desc")
        _, rem = divmod(f, g)
        # Normalize to exactly R bits, MSB-first
        if rem.degree >= 0:
            rvec = GF2(rem.coeffs)
        else:
            rvec = GF2.Zeros(1)
        pad = R - len(rvec)
        if pad > 0:
            rvec = GF2(np.hstack((GF2.Zeros(pad), rvec)))
        elif pad < 0:
            rvec = rvec[-R:]
        return rvec

class Decoder:
    """Syndrome decoder (coset-leader) over GF(2).

    Builds the coset-leader table once.
    """

    def __init__(self, code : LinearCode):
        self.code = code
        self.coset = self.compute_coset_leaders()
        self.S_inv = gf2_inv(code.S)

    def decode(self, noisy_plain: ArrayLike) -> galois.FieldArray:
        # Undo permutation, compute syndrome
        plaintext_tag = noisy_plain @ self.code.P.T
        syndrome = plaintext_tag @ self.code.H.T
        key = tuple(syndrome.tolist())
        # Graceful handling for unknown syndromes (should be rare when t is respected)
        e_hat = self.coset.get(key, GF2.Zeros(self.code.n))
        u_tag = plaintext_tag + e_hat
        m_tag = u_tag[: self.code.k]
        m_hat = m_tag @ self.S_inv
        return m_hat

    def compute_coset_leaders(self) -> Dict[Tuple[int, ...], GF2]:
        """
        Compute coset leaders (minimum weight error patterns) for all syndromes.

        Returns:
            Dict[Tuple[int, ...], GF2]:
                Mapping syndrome → error pattern (coset leader).
                - syndrome is stored as a tuple of ints (0/1).
                - error pattern is a GF(2) vector of length n.
        """
        H = self.code.H
        n = int(self.code.n)
        t = int(self.code.max_errors_num)
        leaders: Dict[Tuple[int, ...], GF2] = {}

        # Precompute the columns of H, since syndrome = XOR of the columns
        # corresponding to error positions.
        H_cols = [H[:, i] for i in range(n)]

        # Enumerate all error patterns up to weight t
        # (we assume the code can correct up to t errors).
        for w in range(0, t + 1):
            for pos in combinations(range(n), w):
                # Syndrome is the XOR (GF(2) sum) of the selected columns of H
                s = GF2.Zeros(H.shape[0])
                for i in pos:
                    s = s + H_cols[i]  # GF(2) addition = XOR

                key = tuple(s.tolist())  # convert syndrome to a hashable tuple

                # If this syndrome has not yet been assigned,
                # the current error pattern is the minimum-weight representative.
                if key not in leaders:
                    e = GF2.Zeros(n)
                    e[list(pos)] = 1
                    leaders[key] = e

        return leaders




# ================================
# Examples (unchanged matrices)
# ================================
# -------------------------------- Golay [23, 12, 7] --------------------------------
A_1 = GF2([
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

S_1 = GF2([
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
    [1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0]
])
P_1 = GF2([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
])
Golay_23_12_7 = [A_1, S_1, P_1]

# -------------------------------- Hamming [5,2,3] --------------------------------
# based on example 1 Lect.13 p.4
A_0 = GF2([[1, 1, 1], [1, 1, 0]])
S_0 = GF2([[0, 1], [1, 0]])
P_0 = GF2([
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0]
])
Hamming_5_2_3 = [A_0, S_0, P_0]

#-------------------------------- Hamming [7,4,3] --------------------------------
# based on example 1 Lect.13 p.4
A_4 = GF2([
    [ 1, 1, 0],
    [ 1, 0, 1],
    [ 0, 1, 1],
    [ 1, 1, 1],
])
S_4 = GF2([
    [1,0,0,0],
    [1,1,0,0],
    [0,1,1,0],
    [0,0,1,1],
])
P_4 = GF2([
    [0,0,0,0,0,1,0],
    [0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0],
    [0,1,0,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,0,1,0,0],
])
Hamming_7_4_3 = [A_4, S_4, P_4]

# Hamming matrix can fix 5 errors. should find P_31X31 $ S_11X11
#this example not completed
A_5= GF2([
        [1, 0, 1 ,1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0 ,1 ,1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0 ,1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1, 0, 0, 1 ,1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
])





LinearCode_dict = {"Golay(n=23,k=12,d=7)": Golay_23_12_7,
                   "Hamming(n=7,k=4,d=3)": Hamming_7_4_3,
                   "Hamming(n=5,k=2,d=3)": Hamming_5_2_3,
                   }


if __name__ == "__main__":
    # Choose which examples to run (kept default to Golay_23_12_7 as before)
    TEST= LinearCode_dict ["Golay(n=23,k=12,d=7)"]

    # GUI-configurable: number of trials per example
    num_messages = 30  # "tests number" in GUI

    # GUI-configurable: channel error probability
    error_prob_channel = 0.05
    if error_prob_channel > 1 or error_prob_channel< 0:
        raise ValueError(
            f" your channel error probability is out of the range: 0-1.")

    # GUI-configurable: number of trials per example
    is_random_message =bool(1)
    # GUI-configurable: Setting parameters for the linear code
    A, S, P = TEST
    code = LinearCode(A, S, P, "Golay(n=23,k=12,d=7)")
    print(f"Linear code parameters: n= {code.n}, k= {code.k}, d_min= {code.d_min}")
    # the message length without CRC
    k_tag = code.k- CRC.generator_degree
    if k_tag<=0 :
        raise ValueError(f" The message is too short. it can't contain the CRC bits. K= {code.k} < CRC length = {CRC.generator_degree}" )

    # GUI-configurable: requested num of errors by the sender
    errors_num = 3
    if errors_num > code.max_errors_num :
        raise ValueError(f" you want to inject {errors_num} errors. it's too big. the max number you can inject is {code.max_errors_num}" )
    # the number of successes in decoding
    decoding_successes_tot = 0
    # the number of successes of crc check
    CRC_successes_tot = 0
    # the total number of trails
    trials_total_num = 0

    decoder= Decoder(code)

    #🔹 Call the grid test (uses Hamming(7,4,3) internally)
    test_sweep_simulation_grid()











