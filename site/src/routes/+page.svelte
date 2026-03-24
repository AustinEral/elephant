<script lang="ts">
	import { base } from '$app/paths';
	import SiteNav from '$lib/components/SiteNav.svelte';
	import { benchmarkReadmeUrl, docsUrl, protocolUrl, repoUrl, resultCardUrl } from '$lib/site';

	const heroFacts = [
		'Rust service',
		'REST + MCP',
		'PostgreSQL + pgvector',
		'Fact extraction'
	] as const;

	const whyPoints = [
		{
			label: 'Memory starts with facts',
			detail:
				'Facts, relationships, and preferences should be stored directly. That gives the system something concrete to retrieve and reason over.'
		},
		{
			label: 'Identity needs continuity',
			detail:
				'An agent becomes more coherent when it carries forward what it has learned, what it has said, and what it has done. Memory gives that identity continuity across sessions.'
		},
		{
			label: 'Opinions change',
			detail:
				'Good memory should adapt when new evidence conflicts with old beliefs or preferences. That lets the system track change instead of freezing a stale view.'
		},
		{
			label: 'Users should not repeat themselves',
			detail:
				'When an agent carries details forward across conversations, users spend less time restating context and correcting the same mistakes.'
		}
	] as const;

	const pipelineStages = [
		{
			id: '01',
			name: 'Retain',
			detail: 'Extract facts, events, and preferences from conversation turns.'
		},
		{
			id: '02',
			name: 'Resolve',
			detail: 'Link references to the same people, projects, and organizations across sessions.'
		},
		{
			id: '03',
			name: 'Consolidate',
			detail: 'Merge related facts into observations and update opinions as new evidence arrives.'
		},
		{
			id: '04',
			name: 'Retrieve',
			detail: 'Search memory by meaning, keywords, relationships, and time.'
		},
		{
			id: '05',
			name: 'Reflect',
			detail: 'Generate an answer from the retrieved record.'
		}
	] as const;

	const heroImageUrl = `${base}/hero-elephant.png`;
</script>

<svelte:head>
	<meta
		property="og:title"
		content="Elephant | Long-term memory infrastructure for AI agents"
	/>
	<meta
		property="og:description"
		content="Persistent memory for AI agents with fact extraction, cross-session entity resolution, retrieval, and answer-time reflection."
	/>
</svelte:head>

<div id="top">
	<SiteNav />
</div>

<section class="hero-shell" aria-labelledby="hero-title">
	<div class="hero-topline">
		<p class="hero-kicker">Elephant / long-term memory infrastructure</p>
		<div class="hero-proof-inline">
			<span>LoCoMo benchmark</span>
			<strong>91.2%</strong>
		</div>
	</div>

	<div class="hero-stage">
		<div class="hero-copy">
			<h1 id="hero-title">
				<span class="hero-line-primary">Persistent memory</span>
				<span class="hero-line-secondary">for AI agents</span>
			</h1>
			<p class="hero-body">
				Elephant stores facts, relationships, and events from conversations so agents can answer
				from memory across sessions.
			</p>
		</div>

		<div class="hero-art" aria-hidden="true">
			<img src={heroImageUrl} alt="" />
		</div>
	</div>

	<div class="hero-actions">
		<a class="button button-primary" href={repoUrl} target="_blank" rel="noreferrer"
			>View the repo</a
		>
		<a class="button button-secondary" href={docsUrl} target="_blank" rel="noreferrer"
			>Read the docs</a
		>
	</div>

	<div class="hero-divider" aria-hidden="true"></div>

	<div class="hero-foot">
		<ul class="hero-facts" aria-label="Elephant facts">
			{#each heroFacts as fact}
				<li>{fact}</li>
			{/each}
		</ul>

		<div class="hero-benchmark">
			<div class="hero-benchmark-label">Benchmark evidence</div>
			<div class="hero-links">
				<a href={benchmarkReadmeUrl} target="_blank" rel="noreferrer">LoCoMo README</a>
				<a href={protocolUrl} target="_blank" rel="noreferrer">Benchmark protocol</a>
				<a href={resultCardUrl} target="_blank" rel="noreferrer">Result card</a>
			</div>
		</div>
	</div>
</section>

<section class="why-shell" aria-labelledby="why-title">
	<div class="section-kicker">Why Elephant</div>

	<div class="why-grid">
		<div class="why-copy">
			<h2 id="why-title">Memory needs structure</h2>
			<p>
				Agents work better when memory is stored as facts, identities, and timelines. That gives
				the system a record it can retrieve and reason over as conversations continue.
			</p>
		</div>

		<div class="why-list" role="list">
			{#each whyPoints as point}
				<div class="why-item" role="listitem">
					<div class="why-label">{point.label}</div>
					<p>{point.detail}</p>
				</div>
			{/each}
		</div>
	</div>
</section>

<section class="mechanics-shell" aria-labelledby="mechanics-title">
	<div class="section-kicker">How it works</div>

	<div class="mechanics-copy">
		<h2 id="mechanics-title">A memory pipeline</h2>
		<p>
			Elephant ingests conversation history, stores memory in structured form, retrieves by
			meaning, keywords, relationships, and time, and answers from the retrieved record.
		</p>
	</div>

	<div class="mechanics-strip" role="list" aria-label="Elephant pipeline stages">
		{#each pipelineStages as stage}
			<div class="mechanics-stage" role="listitem">
				<div class="mechanics-stage-id">{stage.id}</div>
				<div class="mechanics-stage-name">{stage.name}</div>
				<p>{stage.detail}</p>
			</div>
		{/each}
	</div>
</section>
