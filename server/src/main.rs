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
             On-chain         Provide --url to play against the OAB PolkaVM contract.\n\n\
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
    set: u16,

    /// Ethereum JSON-RPC endpoint (enables on-chain mode, e.g. http://localhost:8545)
    #[arg(long)]
    url: Option<String>,

    /// Deployed contract address (defaults to deps/open-auto-battler/contract/deployment.json)
    #[arg(long)]
    contract: Option<String>,

    /// `from` address for transactions (defaults to first dev account from eth_accounts).
    /// The node-managed dev account signs the transactions.
    #[arg(long)]
    from: Option<String>,
}

fn main() {
    let args = Args::parse();

    let backend = if let Some(url) = &args.url {
        // On-chain mode
        #[cfg(feature = "chain")]
        {
            eprintln!("Starting on-chain mode...");
            match chain::ChainGameSession::new(
                url,
                args.contract.as_deref(),
                args.from.as_deref(),
                args.set,
            ) {
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
