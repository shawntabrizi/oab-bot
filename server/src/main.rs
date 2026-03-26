//! OAB Server — HTTP game server for AI agents to play Open Auto Battler.

mod chain;
mod constructed;
mod http;
mod local;
mod types;

use std::process;

use clap::Parser;
use http::{Backend, LocalBackend};

#[derive(Parser)]
#[command(
    name = "oab-server",
    about = "HTTP game server for AI agents to play Open Auto Battler.\n\n\
             Modes:\n  \
             Local (default)  Caller provides opponent boards each round.\n  \
             On-chain         Provide --url and --key to play on a live blockchain.\n\n\
             Endpoints:\n  \
             POST /reset   Start new game\n  \
             POST /submit  Submit actions\n  \
             GET  /state   Get current game state\n  \
             GET  /cards   List all cards\n  \
             GET  /sets    List available card sets"
)]
struct Args {
    /// Server port
    #[arg(short, long, default_value_t = 3000)]
    port: u16,

    /// Default card set ID
    #[arg(long, default_value_t = 0)]
    set: u32,

    /// Chain RPC endpoint (enables on-chain mode)
    #[arg(long)]
    url: Option<String>,

    /// Secret key/SURI for signing (or set OAB_SECRET_SEED env var)
    #[arg(long, env = "OAB_SECRET_SEED")]
    key: Option<String>,

    /// Fund accounts via sudo and exit (--key must be sudo key)
    #[arg(long, num_args = 1..)]
    fund: Vec<String>,
}

fn main() {
    let args = Args::parse();

    // Fund mode: fund target accounts and exit
    if !args.fund.is_empty() {
        #[cfg(feature = "chain")]
        {
            let url = args.url.as_deref().unwrap_or_else(|| {
                eprintln!("Error: --url required for --fund mode.");
                process::exit(1);
            });
            let key = args.key.as_deref().unwrap_or_else(|| {
                eprintln!("Error: --key required for --fund mode (must be sudo key).");
                process::exit(1);
            });
            eprintln!("Funding {} accounts...", args.fund.len());
            if let Err(e) = chain::fund_accounts(url, key, &args.fund) {
                eprintln!("Error funding accounts: {}", e);
                process::exit(1);
            }
            eprintln!("All accounts funded.");
            process::exit(0);
        }
        #[cfg(not(feature = "chain"))]
        {
            eprintln!("Error: --fund requires the 'chain' feature.");
            process::exit(1);
        }
    }

    let backend = if let Some(url) = &args.url {
        // On-chain mode
        #[cfg(feature = "chain")]
        {
            let key = match &args.key {
                Some(k) => k.clone(),
                None => {
                    eprintln!("Error: --key or OAB_SECRET_SEED required for on-chain mode.");
                    process::exit(1);
                }
            };

            eprintln!("Starting on-chain mode...");
            match chain::ChainGameSession::new(url, &key, args.set) {
                Ok(session) => Backend::Chain(session),
                Err(e) => {
                    eprintln!("Error: {}", e);
                    process::exit(1);
                }
            }
        }

        #[cfg(not(feature = "chain"))]
        {
            let _ = url;
            eprintln!("Error: Chain mode requires the 'chain' feature.");
            eprintln!("Build with: cargo build -p oab-server");
            process::exit(1);
        }
    } else {
        // Local mode — sessions created on demand via POST /reset
        eprintln!("Starting local mode (default set={})...", args.set);
        Backend::Local(LocalBackend::new(args.set))
    };

    if let Err(e) = http::serve(args.port, backend) {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}
