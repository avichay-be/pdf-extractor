"""
Classify transactions as DEBIT or CREDIT based on cumulative balance calculation.

Strategy:
For each row n in the table:
- balance[n] = balance[n-1] + credit - debit
- If balance[n] ≈ balance[n-1] + amount[n] → amount is CREDIT
- If balance[n] ≈ balance[n-1] - amount[n] → amount is DEBIT

This works for bank statements where there's a running balance column.

Usage:
    python classify_debit_credit.py <input_pdf> [output_md]
"""
import asyncio
import sys
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

from src.services.mistral_client import MistralDocumentClient
from config import settings

load_dotenv()


def parse_number(text):
    """
    Parse a number from text, handling various formats.

    Examples:
        "1,234.56" → 1234.56
        "1.234,56" → 1234.56
        "(123.45)" → -123.45 (parentheses indicate negative)
        "₪1,234" → 1234
    """
    if not text or not isinstance(text, str):
        return None

    # Remove currency symbols and whitespace
    text = text.strip().replace('₪', '').replace('$', '').replace('€', '').strip()

    # Handle parentheses as negative
    is_negative = text.startswith('(') and text.endswith(')')
    if is_negative:
        text = text[1:-1]

    # Remove all non-digit characters except . and ,
    cleaned = re.sub(r'[^\d.,\-]', '', text)

    if not cleaned:
        return None

    # Determine decimal separator
    # If has both . and ,, the last one is decimal separator
    if '.' in cleaned and ',' in cleaned:
        if cleaned.rindex('.') > cleaned.rindex(','):
            # . is decimal separator (e.g., "1,234.56")
            cleaned = cleaned.replace(',', '')
        else:
            # , is decimal separator (e.g., "1.234,56")
            cleaned = cleaned.replace('.', '').replace(',', '.')
    elif ',' in cleaned:
        # Check if it's thousands separator or decimal
        # If only one comma and it's near the end, it's decimal
        comma_pos = cleaned.index(',')
        if comma_pos > len(cleaned) - 4:  # Within last 3 chars
            cleaned = cleaned.replace(',', '.')
        else:
            cleaned = cleaned.replace(',', '')

    try:
        value = float(cleaned)
        return -value if is_negative else value
    except ValueError:
        return None


def find_balance_column(df):
    """
    Find the column that contains the running balance.

    Looks for keywords like "balance", "יתרה", "saldo"
    """
    balance_keywords = ['balance', 'יתרה', 'saldo', 'solde', 'reste', 'ח"שב']

    for col in df.columns:
        col_str = str(col).lower()
        for keyword in balance_keywords:
            if keyword in col_str:
                return col

    # If not found by name, look for column with mostly numeric increasing/decreasing values
    # This is the balance column
    for col in df.columns:
        # Convert to numeric, counting successful conversions
        values = df[col].apply(parse_number)
        numeric_count = values.notna().sum()

        if numeric_count > len(df) * 0.7:  # More than 70% are numbers
            # Check if it looks like a cumulative balance (gradual changes)
            diffs = values.diff().abs()
            avg_diff = diffs.mean()
            max_diff = diffs.max()

            # Balance should have reasonable incremental changes
            if avg_diff > 0 and max_diff < avg_diff * 100:
                return col

    return None


def find_amount_columns(df):
    """
    Find columns that contain transaction amounts.

    Looks for columns with monetary values that aren't the balance.
    Returns dict with 'debit' and 'credit' columns (can be same column).
    """
    amount_keywords = {
        'debit': ['debit', 'חובה', 'withdrawal', 'retrait', 'expense'],
        'credit': ['credit', 'זכות', 'deposit', 'dépôt', 'income']
    }

    result = {'debit': None, 'credit': None, 'amount': None}

    # Look for explicitly named columns
    for col in df.columns:
        col_str = str(col).lower()

        for keyword in amount_keywords['debit']:
            if keyword in col_str:
                result['debit'] = col
                break

        for keyword in amount_keywords['credit']:
            if keyword in col_str:
                result['credit'] = col
                break

    # If not found, look for columns with numbers (excluding balance)
    balance_col = find_balance_column(df)

    for col in df.columns:
        if col == balance_col:
            continue

        # Count numeric values
        values = df[col].apply(parse_number)
        numeric_count = values.notna().sum()

        if numeric_count > len(df) * 0.5:  # More than 50% are numbers
            if result['amount'] is None:
                result['amount'] = col

    return result


def classify_transactions(df):
    """
    Classify transactions as DEBIT or CREDIT based on balance calculation.

    For each row n:
        balance[n] = balance[n-1] + amount[n]  → CREDIT
        balance[n] = balance[n-1] - amount[n]  → DEBIT

    Returns new DataFrame with added classification columns.
    """
    df_copy = df.copy()

    # Find balance column
    balance_col = find_balance_column(df_copy)
    if not balance_col:
        print("Warning: Could not find balance column")
        return df_copy

    print(f"Found balance column: {balance_col}")

    # Find amount columns
    amount_cols = find_amount_columns(df_copy)
    print(f"Found amount columns: {amount_cols}")

    # Parse balance column
    df_copy['_balance_numeric'] = df_copy[balance_col].apply(parse_number)

    # Initialize classification columns
    df_copy['Transaction_Type'] = ''
    df_copy['Classified_Amount'] = None
    df_copy['Balance_Check'] = ''

    # Process each row
    for i in range(1, len(df_copy)):  # Start from 1 (need previous balance)
        prev_balance = df_copy.loc[i-1, '_balance_numeric']
        curr_balance = df_copy.loc[i, '_balance_numeric']

        if prev_balance is None or curr_balance is None:
            continue

        # Try to find the amount
        amount = None

        # Check if we have separate debit/credit columns
        if amount_cols['debit'] and amount_cols['credit']:
            debit_val = parse_number(str(df_copy.loc[i, amount_cols['debit']]))
            credit_val = parse_number(str(df_copy.loc[i, amount_cols['credit']]))

            if debit_val and debit_val > 0:
                amount = debit_val
                df_copy.loc[i, 'Transaction_Type'] = 'DEBIT'
            elif credit_val and credit_val > 0:
                amount = credit_val
                df_copy.loc[i, 'Transaction_Type'] = 'CREDIT'

        # Check if we have a combined amount column
        elif amount_cols['amount']:
            amount = parse_number(str(df_copy.loc[i, amount_cols['amount']]))

            if amount is not None:
                # Use balance calculation to determine type
                balance_diff = curr_balance - prev_balance
                tolerance = abs(balance_diff) * 0.01  # 1% tolerance

                # Check if credit (balance increased)
                if abs(balance_diff - amount) < tolerance:
                    df_copy.loc[i, 'Transaction_Type'] = 'CREDIT'
                    df_copy.loc[i, 'Balance_Check'] = f'✓ ({prev_balance:.2f} + {amount:.2f} = {curr_balance:.2f})'

                # Check if debit (balance decreased)
                elif abs(balance_diff + amount) < tolerance:
                    df_copy.loc[i, 'Transaction_Type'] = 'DEBIT'
                    df_copy.loc[i, 'Balance_Check'] = f'✓ ({prev_balance:.2f} - {amount:.2f} = {curr_balance:.2f})'

                else:
                    df_copy.loc[i, 'Transaction_Type'] = 'UNCLEAR'
                    df_copy.loc[i, 'Balance_Check'] = f'? (diff={balance_diff:.2f}, amount={amount:.2f})'

        # Store the classified amount
        if amount is not None:
            df_copy.loc[i, 'Classified_Amount'] = amount

    # Clean up temporary column
    df_copy = df_copy.drop(columns=['_balance_numeric'])

    # Summary statistics
    total_rows = len(df_copy)
    debits = (df_copy['Transaction_Type'] == 'DEBIT').sum()
    credits = (df_copy['Transaction_Type'] == 'CREDIT').sum()
    unclear = (df_copy['Transaction_Type'] == 'UNCLEAR').sum()

    print(f"\nClassification Summary:")
    print(f"  Total rows: {total_rows}")
    print(f"  DEBIT: {debits} ({100*debits/total_rows:.1f}%)")
    print(f"  CREDIT: {credits} ({100*credits/total_rows:.1f}%)")
    print(f"  UNCLEAR: {unclear} ({100*unclear/total_rows:.1f}%)")

    return df_copy


def extract_tables_from_markdown(markdown_content: str):
    """Extract tables from markdown content."""
    tables = []
    lines = markdown_content.split('\n')

    current_table = []
    in_table = False
    table_num = 0

    for i, line in enumerate(lines):
        if '|' in line and line.strip():
            if not in_table:
                in_table = True
                table_num += 1
                current_table = [line]
            else:
                current_table.append(line)
        else:
            if in_table and current_table:
                # Parse table to DataFrame
                try:
                    # Remove separator lines
                    table_lines = [l for l in current_table if not re.match(r'^\s*\|[\s\-:]+\|\s*$', l)]

                    if len(table_lines) > 1:
                        # Parse header
                        header = [cell.strip() for cell in table_lines[0].split('|')[1:-1]]

                        # Parse data rows
                        data = []
                        for line in table_lines[1:]:
                            row = [cell.strip() for cell in line.split('|')[1:-1]]
                            if len(row) == len(header):
                                data.append(row)

                        df = pd.DataFrame(data, columns=header)

                        tables.append({
                            'number': table_num,
                            'df': df,
                            'line_start': i - len(current_table),
                            'line_end': i
                        })
                except Exception as e:
                    print(f"Warning: Could not parse table {table_num}: {e}")

                current_table = []
                in_table = False

    return tables


async def process_pdf_with_classification(pdf_path: str, output_path: str):
    """
    Process PDF with Mistral, extract tables, and classify transactions.
    """
    print(f"\nProcessing: {pdf_path}")
    print(f"Output: {output_path}\n")

    # Step 1: Extract with Mistral
    print("[1/4] Extracting content with Mistral...")
    async with MistralDocumentClient(
        api_key=settings.AZURE_API_KEY,
        api_url=settings.MISTRAL_API_URL,
        model=settings.MISTRAL_MODEL
    ) as mistral_client:
        markdown_content, _ = await mistral_client.process_document(
            pdf_path=pdf_path,
            enable_validation=False
        )

    print(f"  ✓ Extracted {len(markdown_content)} characters")

    # Step 2: Extract tables
    print("\n[2/4] Extracting tables from markdown...")
    tables = extract_tables_from_markdown(markdown_content)
    print(f"  ✓ Found {len(tables)} tables")

    # Step 3: Classify transactions in each table
    print("\n[3/4] Classifying transactions...")
    classified_tables = []

    for table in tables:
        print(f"\n  Table {table['number']}:")
        classified_df = classify_transactions(table['df'])
        classified_tables.append({
            **table,
            'classified_df': classified_df
        })

    # Step 4: Save results
    print("\n[4/4] Saving classified tables...")

    markdown_output = f"# Classified Transactions from {Path(pdf_path).name}\n\n"
    markdown_output += f"**Extraction Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    markdown_output += f"**Method:** Mistral + Balance-based Classification\n\n"
    markdown_output += "---\n\n"

    for table in classified_tables:
        markdown_output += f"## Table {table['number']}\n\n"
        markdown_output += table['classified_df'].to_markdown(index=False)
        markdown_output += "\n\n---\n\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_output)

    print(f"  ✓ Saved to: {output_path}")

    return {
        'tables': len(classified_tables),
        'total_debits': sum((t['classified_df']['Transaction_Type'] == 'DEBIT').sum() for t in classified_tables),
        'total_credits': sum((t['classified_df']['Transaction_Type'] == 'CREDIT').sum() for t in classified_tables)
    }


async def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_debit_credit.py <input_pdf> [output_md]")
        print("\nExample:")
        print("  python classify_debit_credit.py data/bank_statements.pdf output/classified.md")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else f"{Path(pdf_path).stem}_classified.md"

    stats = await process_pdf_with_classification(pdf_path, output_path)

    print(f"\n✓ Done!")
    print(f"  Tables processed: {stats['tables']}")
    print(f"  Total DEBIT transactions: {stats['total_debits']}")
    print(f"  Total CREDIT transactions: {stats['total_credits']}")


if __name__ == "__main__":
    asyncio.run(main())
