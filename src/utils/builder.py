"""Build utilities for Solana smart contracts."""

import subprocess
from pathlib import Path


class Builder:
    """Build utilities for Anchor/Rust Solana projects."""

    def __init__(self, project_path: Path):
        """Initialize with project path.

        Args:
            project_path: Path to the Anchor project directory
        """
        self.project_path = Path(project_path)

    def run_command(
        self,
        cmd: list[str],
        capture_output: bool = False,
        timeout: int | None = 300,
        stream_output: bool = False,
    ) -> tuple[bool, str, str]:
        """Run a shell command in the project directory.

        Args:
            cmd: Command and arguments as list
            capture_output: Whether to capture stdout/stderr (for return value)
            timeout: Command timeout in seconds
            stream_output: Whether to stream output to terminal in real-time

        Returns:
            Tuple of (success, stdout, stderr)
        """
        try:
            if stream_output:
                # Stream output to terminal while still capturing
                # cmd is built from hardcoded strings (anchor, cargo), safe to run
                process = subprocess.Popen(  # noqa: S602,S603
                    cmd,
                    cwd=self.project_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                output_lines = []
                for line in process.stdout:
                    print(line, end="")  # Stream to terminal
                    output_lines.append(line)

                process.wait()
                output = "".join(output_lines) if output_lines else ""
                return process.returncode == 0, output, ""

            result = subprocess.run(  # noqa: S603
                cmd,
                cwd=self.project_path,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                check=False,
            )
            return result.returncode == 0, result.stdout or "", result.stderr or ""
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except FileNotFoundError:
            return False, "", f"Command not found: {cmd[0]}"

    def cargo_check_sbf(self) -> tuple[bool, str]:
        """Run cargo check for SBF target.

        Returns:
            Tuple of (success, output)
        """
        success, stdout, stderr = self.run_command(
            ["cargo", "check", "--target", "sbf-solana-solana"]
        )
        return success, (stdout or "") + (stderr or "")

    def rustfmt(self, file_path: Path | None = None) -> tuple[bool, str]:
        """Run rustfmt on files.

        Args:
            file_path: Specific file to format, or None for all files

        Returns:
            Tuple of (success, output)
        """
        if file_path:
            success, stdout, stderr = self.run_command(["rustfmt", str(file_path)])
        else:
            # Format all Rust files
            success, stdout, stderr = self.run_command(["cargo", "+nightly", "fmt"])
        return success, (stdout or "") + (stderr or "")

    def anchor_build(self, stream: bool = True) -> tuple[bool, str]:
        """Run anchor build.

        Args:
            stream: Whether to stream output to terminal in real-time

        Returns:
            Tuple of (success, output)
        """
        success, stdout, stderr = self.run_command(
            ["anchor", "build"],
            stream_output=stream,
        )
        return success, (stdout or "") + (stderr or "")

    def cargo_build_sbf(self) -> tuple[bool, str]:
        """Run cargo build-sbf.

        Returns:
            Tuple of (success, output)
        """
        success, stdout, stderr = self.run_command(["cargo", "build-sbf"])
        return success, (stdout or "") + (stderr or "")

    def get_build_artifact(self) -> Path | None:
        """Get the build artifact path.

        Returns:
            Path to the compiled program .so file, or None
        """
        # Look for .so files in target/deploy
        deploy_dir = self.project_path / "target" / "deploy"
        if deploy_dir.exists():
            for f in deploy_dir.glob("*.so"):
                return f
        return None

    def verify_build(self, stream: bool = True) -> tuple[bool, str]:
        """Run full build verification.

        Args:
            stream: Whether to stream output to terminal in real-time

        Returns:
            Tuple of (success, output)
        """
        # First try anchor build
        success, output = self.anchor_build(stream=stream)
        if success:
            artifact = self.get_build_artifact()
            if artifact:
                return True, f"Build successful: {artifact}"
            return True, "Build successful (artifact location unknown)"
        return False, output

    def check_prerequisites(self) -> tuple[bool, list[str]]:
        """Check if build prerequisites are available.

        Returns:
            Tuple of (all_available, list of missing tools)
        """
        missing = []
        tools = ["cargo", "rustc", "anchor"]  # anchor is optional

        for tool in tools:
            result = subprocess.run(  # noqa: S603
                ["/usr/bin/which", tool],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                missing.append(tool)

        return len(missing) == 0, missing

    def anchor_init(self, project_name: str) -> tuple[bool, str]:
        """Run anchor init to create fresh Anchor project structure.

        Args:
            project_name: Name for the new project

        Returns:
            Tuple of (success, output)
        """
        success, stdout, stderr = self.run_command(["anchor", "init", project_name, "--no-git"])
        return success, (stdout or "") + (stderr or "")
